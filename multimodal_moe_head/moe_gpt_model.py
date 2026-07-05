"""MoEGPTModel — GPTModel subclass with the two-expert MoE output head.

Drop-in replacement for GPTModel on post-process ranks.  The standard
ColumnParallelLinear output layer is replaced by MoEOutputHead, which routes
tokens to a text expert or a vision/audio expert before computing per-token
cross-entropy loss.

The router auxiliary loss is attached to the autograd graph via
AuxLossAutoScaler — a zero-value-impact custom Function that fires
router_loss.backward(1.0) during the main backward pass without inflating
the reported LM loss scalar.
"""

from typing import Optional

import torch
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.transformer_config import TransformerConfig

from multimodal_moe_head.config import MoEOutputHeadConfig
from multimodal_moe_head.integration import build_moe_output_head


class AuxLossAutoScaler(torch.autograd.Function):
    """Attach an auxiliary loss to the autograd graph without changing the output value.

    During forward: returns `output` unchanged.
    During backward: passes gradients through to `output` normally, and sends
    a gradient of 1.0 to `aux_loss` so it backprops independently.

    This mirrors the MTPLossAutoScaler pattern used by Megatron for multi-token
    prediction auxiliary losses.
    """

    @staticmethod
    def forward(ctx, output: torch.Tensor, aux_loss: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        _aux_loss, = ctx.saved_tensors
        return grad_output, torch.ones_like(_aux_loss)


class MoEGPTModel(GPTModel):
    """GPTModel with the two-expert modality-routed output head.

    The only behavioural difference from GPTModel is the post-process stage:
    instead of a single ColumnParallelLinear projection over the full vocab,
    this model routes each token to a text or vision/audio expert and returns
    the resulting per-token cross-entropy loss [b, s] — the same shape that
    loss_func in pretrain_gpt.py already expects.
    """

    def __init__(
        self,
        *args,
        head_config: MoEOutputHeadConfig,
        **kwargs,
    ):
        # Embedding weights cannot be shared with the MoE head because the
        # experts have different (smaller) vocab sizes.
        kwargs['share_embeddings_and_output_weights'] = False
        super().__init__(*args, **kwargs)

        if self.post_process:
            # Free the full-vocab ColumnParallelLinear that GPTModel just built,
            # then replace it with the two-expert head.
            del self.output_layer
            self.output_layer = build_moe_output_head(
                self.config, head_config, tp_group=self.pg_collection.tp
            )

        # Cached by _postprocess for TensorBoard logging in loss_func.
        self._moe_router_loss: Optional[torch.Tensor] = None
        self._moe_head_stats: Optional[dict] = None

    def _postprocess(
        self,
        hidden_states,
        input_ids,
        position_ids,
        labels,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        mtp_in_postprocess=None,
        loss_mask=None,
        decoder_input=None,
        attention_mask=None,
        inference_params=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        runtime_gather_output=None,
        extra_block_kwargs=None,
        inference_context=None,
        is_spec_decode=None,
    ):
        if not self.post_process:
            return hidden_states

        # MTP + MoE output head is untested: the MoE head returns a per-token
        # [b, s] loss and bypasses the standard output_layer that process_mtp_loss
        # expects, so MTP layers would not be fed correctly. Fail fast instead of
        # silently mistraining.
        assert not getattr(self.config, 'mtp_num_layers', None), (
            "MoEGPTModel does not support Multi-Token Prediction (mtp_num_layers). "
            "Disable MTP to use --moe-output-head."
        )

        from megatron.training import get_args
        current_step = get_args().curr_iteration

        result = self.output_layer(
            hidden_states,
            labels=labels,
            current_step=current_step,
            loss_mask=loss_mask,
        )

        if labels is None:
            # Autoregressive generation is not yet supported for this head.
            # Stage-1 pretraining never reaches this path (labels are always
            # provided during training).
            raise NotImplementedError(
                "MoEGPTModel does not support generation yet. "
                "labels=None is only expected during inference."
            )

        lm_loss = result.lm_loss  # [b, s] per-token cross-entropy

        # Attach the router CE loss to the autograd graph so it backprops
        # alongside the LM loss without changing the reported loss value.
        if result.router_loss is not None:
            coeff = self.output_layer.head_config.router_loss_coeff
            lm_loss = AuxLossAutoScaler.apply(lm_loss, coeff * result.router_loss)

        # Cache for logging in loss_func (detach — we only want the scalar value).
        self._moe_router_loss = (
            result.router_loss.detach() if result.router_loss is not None else None
        )

        # Expert utilization and router accuracy. Each metric is reported as a
        # [numerator, denominator] pair so the cross-rank reduction in training.py
        # (Σnumerator / Σdenominator over the data-parallel group) yields an EXACT
        # token-weighted GLOBAL value, robust to unequal per-rank/micro-batch token
        # counts (e.g. variable padding/masking).
        # flat_labels layout matches head.py: labels [b,s] → transpose → [s,b] → view(-1) → [s*b].
        N = result.routing_decision.numel()
        if N > 0:
            flat_labels = labels.t().contiguous().view(-1)  # [s*b], same order as router_logits
            text_vocab_size = self.output_layer.head_config.text_vocab_size
            gt_modality = (flat_labels >= text_vocab_size).long()  # 0=text, 1=vision/audio
            valid = flat_labels >= 0                               # exclude masked (-100) tokens

            # ---- Learned-router quality metrics ---------------------------------
            # All measured from the LEARNED router (argmax of its own logits) vs the
            # ground-truth modality, ALWAYS — including during teacher forcing, where
            # the actual dispatch is forced to ground truth and would be trivially
            # "100% correct". Computed on-device as [numerator, denominator] pairs
            # (no host sync); the DP-group reduction turns each into an exact
            # token-weighted global value. Confusion counts on valid tokens:
            logits = result.router_logits.float()
            pred_v = logits.argmax(dim=-1)[valid]   # router prediction, valid tokens
            gt_v = gt_modality[valid]               # ground truth,      valid tokens
            is_text = gt_v == 0
            is_va = gt_v == 1
            n_valid = valid.sum().float()
            n_text_gt = is_text.sum().float()
            n_va_gt = is_va.sum().float()
            text_correct = (is_text & (pred_v == 0)).sum().float()   # text routed to text (TP_text)
            va_correct = (is_va & (pred_v == 1)).sum().float()       # VA routed to VA  (TP_va)
            pred_text = (pred_v == 0).sum().float()                  # tokens routed to text expert
            pred_va = (pred_v == 1).sum().float()                    # tokens routed to VA expert
            # Mean max-softmax probability = router decisiveness / confidence.
            conf_sum = torch.softmax(logits, dim=-1).max(dim=-1).values[valid].sum()

            self._moe_head_stats = {
                # Learned-router quality: overall accuracy + per-modality recall
                # (token reaches the right expert) and precision (expert receives
                # only its modality). Together they are the full confusion matrix.
                'moe_router_accuracy': torch.stack([text_correct + va_correct, n_valid]).detach(),
                'moe_router_text_recall': torch.stack([text_correct, n_text_gt]).detach(),
                'moe_router_va_recall': torch.stack([va_correct, n_va_gt]).detach(),
                'moe_router_text_precision': torch.stack([text_correct, pred_text]).detach(),
                'moe_router_va_precision': torch.stack([va_correct, pred_va]).detach(),
                # Learned-router intended load (collapse detector: what the router
                # WOULD route to VA, independent of the teacher-forced dispatch above).
                'moe_router_pred_va_frac': torch.stack([pred_va, n_valid]).detach(),
                # Mean confidence in its own top choice (sharpening over training).
                'moe_router_confidence': torch.stack([conf_sum, n_valid]).detach(),
            }
        else:
            self._moe_head_stats = None

        return lm_loss  # [b, s] — same format as standard GPTModel training return

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Sharded state dict that tolerates the MoE output head.

        GPTModel -> LanguageModule.sharded_state_dict assumes a single
        `output_layer.weight` tensor (to tie with embeddings or to mark
        allow_shape_mismatch). Our head replaces that with
        `output_layer.router.weight` and `output_layer.experts.projections.*`,
        so that code path raises KeyError('output_layer.weight').

        We bypass LanguageModule's output-layer logic by calling MegatronModule's
        default recursive implementation directly (which produces the correct
        MoE-head keys), then reapply GPTModel's only other relevant step: dropping
        the legacy `output_layer._extra_state`. Embedding/output weights are never
        tied here (share_embeddings_and_output_weights is forced False), and MTP
        is unused, so nothing else from the overridden methods is needed.
        """
        from megatron.core.transformer.module import MegatronModule

        sharded_state_dict = MegatronModule.sharded_state_dict(
            self, prefix, sharded_offsets, metadata
        )

        output_layer_extra_state_key = f'{prefix}output_layer._extra_state'
        output_extra_state = sharded_state_dict.pop(output_layer_extra_state_key, None)
        assert not (
            output_extra_state and output_extra_state.data
        ), f'Expected output layer extra state to be empty, got: {output_extra_state}'

        return sharded_state_dict
