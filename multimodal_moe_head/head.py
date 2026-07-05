from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.distributed as dist
from torch import nn

from megatron.core import parallel_state, tensor_parallel
from megatron.core.tensor_parallel.mappings import all_to_all as _megatron_all_to_all
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

from multimodal_moe_head.config import MoEOutputHeadConfig
from multimodal_moe_head.experts import OutputProjectionExperts
from multimodal_moe_head.router_loss import (
    modality_target_from_labels,
    router_modality_loss,
    teacher_forced_routing_map,
)


@dataclass
class MoEOutputHeadResult:
    """Outputs returned by `MoEOutputHead.forward`.

    Per-token tensors index the flattened sequence × batch (`N = s*b`).
    Loss tensors are populated only when `labels` is provided.

    EP=2 note: `text_logits` / `va_logits` hold the logits computed by the
    *local* expert only. EP rank 0 (text expert) has `text_logits` set and
    `va_logits=None`; EP rank 1 (VA expert) has the opposite. This is
    intentional — the output head does not all-to-all the logits back because
    they have different vocab sizes.
    """

    router_logits: torch.Tensor                 # [N, 2]
    routing_decision: torch.Tensor              # [N] long in {0, 1}

    text_logits: Optional[torch.Tensor]         # [N_text, V_text/tp_size] or None
    va_logits: Optional[torch.Tensor]           # [N_va,   V_va/tp_size]   or None
    text_token_indices: Optional[torch.Tensor]  # [N_text] indices into the flattened sequence
    va_token_indices: Optional[torch.Tensor]    # [N_va]

    lm_loss: Optional[torch.Tensor] = None              # [b, s], filled per-token
    router_loss: Optional[torch.Tensor] = None          # scalar
    total_loss: Optional[torch.Tensor] = None           # scalar


class MoEOutputHead(MegatronModule):
    """Two-expert modality-routed output head.

    Supports two expert-parallel configurations:

    EP=1 (replicated)
        Both experts reside on every rank. Each rank processes all tokens
        locally. This is the default, safest configuration.

    EP=2 (one expert per rank)
        Expert 0 (text projection) lives on EP rank 0; expert 1 (VA
        projection) lives on EP rank 1. Tokens are exchanged via a pair of
        All-to-All collectives — one for the forward dispatch of hidden states
        and labels, one for the reverse combine of per-token LM losses.

        With world_size=4 (TP=1, PP=1, EP=2, DP=2) two EP groups exist:
        [0,1] and [2,3]. Each of the 4 ranks carries its own micro-batch
        (DP=4 for non-expert layers, DP=2 per expert). Within each EP group
        the dispatch A2A routes text tokens from all ranks to the text expert
        and VA tokens from all ranks to the VA expert. After the combine A2A
        every rank reconstructs the full per-token loss for its own tokens.
        The router is replicated on all ranks.

    Inputs
    ------
    hidden_states : [s, b, H]
        Final hidden states after the transformer's last layer norm.
    labels : [b, s], optional
        Target token ids. When provided the head computes per-token LM loss,
        router CE loss, optional load-balance penalty, and a combined scalar
        `total_loss`.

    Routing
    -------
    Training: if `current_step < config.teacher_force_steps`, routing is
    forced to the ground-truth modality from labels. Otherwise argmax of
    router logits.  Inference (`labels is None`): always argmax.
    """

    def __init__(
        self,
        config: TransformerConfig,
        head_config: MoEOutputHeadConfig,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(config=config)
        self.head_config = head_config
        self._ep = head_config.expert_parallel_size

        self.text_vocab_size = head_config.text_vocab_size
        self.va_vocab_size = head_config.va_vocab_size

        self.router = nn.Linear(config.hidden_size, 2, bias=False)
        self.router.weight.data = self.router.weight.data.to(dtype=config.params_dtype)

        if self._ep == 1:
            local_expert_indices: List[int] = [0, 1]
            self._local_expert_idx: Optional[int] = None
        else:
            # EP=2: each rank owns exactly one expert.
            ep_rank = parallel_state.get_expert_model_parallel_rank()
            self._local_expert_idx = ep_rank  # 0 = text, 1 = VA
            local_expert_indices = [ep_rank]

        self.experts = OutputProjectionExperts(
            config=config,
            vocab_sizes=head_config.expert_vocab_sizes,
            local_expert_indices=local_expert_indices,
            tp_group=tp_group,
            expert_parallel=(self._ep > 1),
        )

        # Under EP>1 each rank holds a *different* expert (text on rank 0, VA on
        # rank 1) with different vocab shapes.  Tag these weights so Megatron's
        # DDP / distributed optimizer reduce their gradients over the
        # expert-data-parallel group (size 1 here) instead of the dense DP group
        # — otherwise it would try to all-reduce mismatched shapes across ranks
        # and hang.  Megatron reads `param.allreduce`: False => expert-parallel.
        # Under EP=1 the experts are replicated and must stay on the dense path
        # (allreduce defaults to True), so we only tag when EP>1.
        if self._ep > 1:
            for proj in self.experts.projections.values():
                for param in proj.parameters():
                    setattr(param, 'allreduce', False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_router(self, flat_hidden: torch.Tensor) -> torch.Tensor:
        # The router weight lives in the model's param dtype (e.g. bf16 under
        # --bf16). Match the input dtype to the weight for the matmul, then
        # upcast the logits to float32 for numerically-stable CE loss / argmax
        # downstream. Casting the input to float32 directly would mismatch a
        # bf16 weight (the standalone tests only ran in float32, so missed this).
        logits = self.router(flat_hidden.to(self.router.weight.dtype))
        return logits.float()

    # --- EP=2 helpers -------------------------------------------------

    @staticmethod
    def _ep2_exchange_splits(
        text_count: int,
        va_count: int,
        ep_rank: int,
        ep_group: torch.distributed.ProcessGroup,
        device: torch.device,
    ):
        """All-reduce a 2×2 matrix to learn how many tokens each rank sends
        to each destination rank.

        Returns
        -------
        input_splits : list[int]  — tokens sent from this rank to each EP rank
        output_splits : list[int] — tokens received by this rank from each EP rank
        """
        # Row ep_rank encodes what this rank sends: [text_count, va_count].
        all_splits = torch.zeros(2, 2, dtype=torch.long, device=device)
        all_splits[ep_rank, 0] = text_count
        all_splits[ep_rank, 1] = va_count
        dist.all_reduce(all_splits, op=dist.ReduceOp.SUM, group=ep_group)
        # output_splits[i] = what rank i sends *to me* = all_splits[i, ep_rank]
        output_splits = [int(all_splits[i, ep_rank].item()) for i in range(2)]
        input_splits = [text_count, va_count]
        return input_splits, output_splits

    @staticmethod
    def _ep2_dispatch_hidden(
        flat_hidden: torch.Tensor,
        text_indices: torch.Tensor,
        va_indices: torch.Tensor,
        input_splits: List[int],
        output_splits: List[int],
        ep_group: torch.distributed.ProcessGroup,
    ) -> torch.Tensor:
        """Permute hidden states and all-to-all dispatch to expert ranks.

        Tokens routed to expert 0 (text) go to EP rank 0;
        tokens routed to expert 1 (VA) go to EP rank 1.
        The all_to_all call is wrapped in Megatron's autograd function so
        gradients flow back correctly via the reverse A2A.
        """
        # Text tokens first, VA tokens second — matching how output_splits
        # are laid out (rank 0 receives from index 0, rank 1 from index 1).
        permuted = torch.cat([flat_hidden[text_indices], flat_hidden[va_indices]], dim=0)
        return _megatron_all_to_all(ep_group, permuted, output_splits, input_splits)

    @staticmethod
    def _ep2_dispatch_labels(
        flat_labels: torch.Tensor,
        text_indices: torch.Tensor,
        va_indices: torch.Tensor,
        input_splits: List[int],
        output_splits: List[int],
        ep_group: torch.distributed.ProcessGroup,
    ) -> torch.Tensor:
        """Permute and all-to-all dispatch integer labels (not differentiable)."""
        permuted = torch.cat(
            [flat_labels[text_indices], flat_labels[va_indices]]
        ).contiguous()
        out_count = sum(output_splits)
        global_labels = torch.empty(
            out_count, dtype=flat_labels.dtype, device=flat_labels.device
        )
        dist.all_to_all_single(
            global_labels,
            permuted,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=ep_group,
        )
        return global_labels

    @staticmethod
    def _ep2_combine_loss(
        per_expert_loss: torch.Tensor,
        input_splits: List[int],
        output_splits: List[int],
        ep_group: torch.distributed.ProcessGroup,
    ) -> torch.Tensor:
        """Reverse A2A: send per-expert scalar losses back to their origin ranks.

        After this call every EP rank holds a [N] float32 tensor of per-token
        LM losses in *permuted* order (text tokens first, VA tokens second).
        The caller scatters them back to their original positions.
        """
        # Combine direction swaps output_split_sizes and input_split_sizes
        # relative to the dispatch direction.
        return _megatron_all_to_all(
            ep_group, per_expert_loss.float(), input_splits, output_splits
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _log_router_inputs(self, labels: torch.Tensor, current_step: int) -> None:
        """Print the per-sample modality of this rank's micro-batch.

        Logs the first `head_config.log_router_inputs` micro-batches seen on this
        rank. With --use-packed-seq-params the MBS samples are packed into a
        single [b=1, s=MBS*seq_len] row, so we split each row into seq_len-sized
        chunks (the original samples) and classify each: a text-blend sample is
        ~all text ids (< text_vocab_size); a vision-blend sample (img2txt /
        interleave) mixes caption text with image/audio ids (>= text_vocab_size).
        This makes the 1-text + 1-vision interleaving directly verifiable per rank.
        """
        count = getattr(self, "_router_input_log_count", 0)
        if count >= self.head_config.log_router_inputs:
            return
        self._router_input_log_count = count + 1

        tvs = self.text_vocab_size
        try:
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else 0
        except Exception:
            rank = 0
        try:
            ep_rank = parallel_state.get_expert_model_parallel_rank()
            dp_rank = parallel_state.get_data_parallel_rank()
        except Exception:
            ep_rank = dp_rank = -1
        # Per-sample length so we can un-pack the packed micro-batch.
        try:
            from megatron.training import get_args
            seqlen = int(get_args().seq_length)
        except Exception:
            seqlen = labels.shape[-1]

        parts = []
        flat = labels.reshape(-1)
        n_sub = max(1, flat.numel() // seqlen)
        for j in range(n_sub):
            sub = flat[j * seqlen:(j + 1) * seqlen]
            va = int((sub >= tvs).sum().item())
            txt = int(((sub >= 0) & (sub < tvs)).sum().item())
            frac = va / max(1, sub.numel())
            kind = "VISION" if frac > 0.5 else ("TEXT" if frac < 0.02 else "MIXED")
            parts.append(f"sample{j}={kind}(text={txt},va={va},va_frac={frac:.2f})")
        print(
            f"[ROUTER-INPUT] iter={current_step} rank={rank} ep_rank={ep_rank} "
            f"dp_rank={dp_rank} packed_rows={labels.shape[0]} samples={n_sub} "
            + " ".join(parts),
            flush=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        current_step: int = 0,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> MoEOutputHeadResult:
        s, b, h = hidden_states.shape
        N = s * b
        flat_hidden = hidden_states.reshape(N, h)

        router_logits = self._compute_router(flat_hidden)
        device = flat_hidden.device

        training_with_labels = labels is not None

        # ---- routing map -----------------------------------------------
        if training_with_labels:
            labels_sb = labels.transpose(0, 1).contiguous()
            flat_labels = labels_sb.reshape(N)
            modality_target = modality_target_from_labels(flat_labels, self.text_vocab_size)
            if self.head_config.log_router_inputs > 0:
                self._log_router_inputs(labels, current_step)
            if current_step < self.head_config.teacher_force_steps:
                routing_map = teacher_forced_routing_map(modality_target)
            else:
                choice = router_logits.argmax(dim=-1)
                routing_map = torch.zeros((N, 2), dtype=torch.bool, device=device)
                routing_map[torch.arange(N, device=device), choice] = True
        else:
            flat_labels = None
            modality_target = None
            choice = router_logits.argmax(dim=-1)
            routing_map = torch.zeros((N, 2), dtype=torch.bool, device=device)
            routing_map[torch.arange(N, device=device), choice] = True

        text_indices = routing_map[:, 0].nonzero(as_tuple=False).squeeze(-1)
        va_indices = routing_map[:, 1].nonzero(as_tuple=False).squeeze(-1)

        # ---- expert forward (EP=1 vs EP=2) ----------------------------
        if self._ep == 1:
            text_hidden = flat_hidden.index_select(0, text_indices)
            va_hidden = flat_hidden.index_select(0, va_indices)
            # ALWAYS run both experts, even on 0 tokens ([0, H] -> [0, V]). Keeps
            # the autograd graph routing-independent so each expert weight's grad
            # fires at the same point in backward on every data-parallel rank —
            # otherwise --overlap-grad-reduce reorders the grad all-reduce and
            # NCCL desyncs (same failure mode as the EP=2 expert-DP path).
            text_logits = self.experts.forward(0, text_hidden)
            va_logits = self.experts.forward(1, va_hidden)
            ep2_state = None
        else:
            ep_group = parallel_state.get_expert_model_parallel_group()
            ep_rank = parallel_state.get_expert_model_parallel_rank()
            text_count = text_indices.numel()
            va_count = va_indices.numel()

            input_splits, output_splits = self._ep2_exchange_splits(
                text_count, va_count, ep_rank, ep_group, device
            )

            global_hidden = self._ep2_dispatch_hidden(
                flat_hidden, text_indices, va_indices,
                input_splits, output_splits, ep_group,
            )

            global_labels_ep2: Optional[torch.Tensor] = None
            if training_with_labels:
                global_labels_ep2 = self._ep2_dispatch_labels(
                    flat_labels, text_indices, va_indices,
                    input_splits, output_splits, ep_group,
                )

            # ALWAYS run the local expert, even when it received zero tokens
            # (global_hidden is [0, H] -> local_logits is [0, V]). Running it
            # unconditionally keeps the autograd graph — and therefore the
            # gradient-reduction collective order under --overlap-grad-reduce —
            # identical across the expert's data-parallel replicas regardless of
            # routing. A routing-dependent branch here makes the expert weight's
            # grad fire at a different point in backward on each replica, which
            # reorders the EXPERT_DATA_PARALLEL_GROUP reduce-scatter and
            # deadlocks NCCL (collective-sequence desync).
            local_logits = self.experts.forward(self._local_expert_idx, global_hidden)

            # Each rank exposes only its own expert's logits.
            if self._local_expert_idx == 0:
                text_logits = local_logits
                va_logits = None
            else:
                text_logits = None
                va_logits = local_logits

            ep2_state = dict(
                ep_group=ep_group,
                local_logits=local_logits,
                global_labels=global_labels_ep2,
                global_hidden=global_hidden,
                input_splits=input_splits,
                output_splits=output_splits,
            )

        # ---- assemble result ------------------------------------------
        result = MoEOutputHeadResult(
            router_logits=router_logits,
            routing_decision=routing_map.long().argmax(dim=-1),
            text_logits=text_logits,
            va_logits=va_logits,
            text_token_indices=text_indices,
            va_token_indices=va_indices,
        )

        if training_with_labels:
            if self._ep == 1:
                self._fill_losses(
                    result=result,
                    flat_labels=flat_labels,
                    modality_target=modality_target,
                    router_logits=router_logits,
                    routing_map=routing_map,
                    s=s,
                    b=b,
                    loss_mask=loss_mask,
                )
            else:
                self._fill_losses_ep2(
                    result=result,
                    flat_labels=flat_labels,
                    modality_target=modality_target,
                    router_logits=router_logits,
                    routing_map=routing_map,
                    text_indices=text_indices,
                    va_indices=va_indices,
                    N=N,
                    s=s,
                    b=b,
                    loss_mask=loss_mask,
                    device=device,
                    **ep2_state,
                )

        return result

    # ------------------------------------------------------------------
    # Loss computation — EP=1
    # ------------------------------------------------------------------

    def _fill_losses(
        self,
        result: MoEOutputHeadResult,
        flat_labels: torch.Tensor,
        modality_target: torch.Tensor,
        router_logits: torch.Tensor,
        routing_map: torch.Tensor,
        s: int,
        b: int,
        loss_mask: Optional[torch.Tensor],
    ) -> None:
        N = s * b
        device = flat_labels.device

        per_token_lm_loss = torch.zeros(N, dtype=torch.float32, device=device)

        # ALWAYS compute CE for both experts (text_logits / va_logits are always
        # tensors now, possibly [0, V]). index_add_ with empty indices is a
        # no-op. Unconditional execution keeps the autograd graph — and the grad
        # all-reduce order under --overlap-grad-reduce — identical across DP
        # ranks regardless of routing.
        text_labels_full = flat_labels.index_select(0, result.text_token_indices)
        text_labels_safe = text_labels_full.clamp(min=0, max=self.text_vocab_size - 1)
        text_loss = tensor_parallel.vocab_parallel_cross_entropy(
            result.text_logits, text_labels_safe
        )
        per_token_lm_loss.index_add_(0, result.text_token_indices, text_loss.float())

        va_labels_full = flat_labels.index_select(0, result.va_token_indices)
        va_labels_adj = va_labels_full - self.text_vocab_size
        va_labels_safe = va_labels_adj.clamp(min=0, max=self.va_vocab_size - 1)
        va_loss = tensor_parallel.vocab_parallel_cross_entropy(
            result.va_logits, va_labels_safe
        )
        per_token_lm_loss.index_add_(0, result.va_token_indices, va_loss.float())

        lm_loss_bs = per_token_lm_loss.view(s, b).transpose(0, 1).contiguous()
        result.lm_loss = lm_loss_bs

        router_loss = router_modality_loss(router_logits, modality_target)
        result.router_loss = router_loss

        if loss_mask is not None:
            mask = loss_mask.float()
            lm_scalar = (lm_loss_bs * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            lm_scalar = lm_loss_bs.mean()

        result.total_loss = lm_scalar + self.head_config.router_loss_coeff * router_loss

    # ------------------------------------------------------------------
    # Loss computation — EP=2
    # ------------------------------------------------------------------

    def _fill_losses_ep2(
        self,
        result: MoEOutputHeadResult,
        flat_labels: torch.Tensor,
        modality_target: torch.Tensor,
        router_logits: torch.Tensor,
        routing_map: torch.Tensor,
        text_indices: torch.Tensor,
        va_indices: torch.Tensor,
        N: int,
        s: int,
        b: int,
        loss_mask: Optional[torch.Tensor],
        device: torch.device,
        ep_group: torch.distributed.ProcessGroup,
        local_logits: Optional[torch.Tensor],
        global_labels: Optional[torch.Tensor],
        global_hidden: torch.Tensor,
        input_splits: List[int],
        output_splits: List[int],
    ) -> None:
        """Compute training losses for the EP=2 path.

        Each rank computes the cross-entropy loss for its own expert's tokens.
        A reverse all-to-all redistributes per-expert losses so that every rank
        reconstructs the full [N] per-token loss vector. Both EP ranks therefore
        end up with identical `lm_loss` and `total_loss` tensors, which is
        required for consistent backward passes across the DP dimension.
        """
        # ---- per-expert loss -------------------------------------------
        # ALWAYS compute CE — even with zero received tokens (local_logits is
        # [0, V], global_labels is [0]), CE returns a [0] loss. Doing it
        # unconditionally keeps the autograd graph routing-independent so the
        # gradient-reduction collectives stay in lockstep across DP replicas
        # (see the note in forward). No anchors are needed: the expert weight is
        # always connected to the loss through this matmul, and global_hidden
        # (hence the dispatch A2A backward) is connected through it too.
        assert global_labels is not None
        if self._local_expert_idx == 0:  # text expert
            labels_adj = global_labels.clamp(min=0, max=self.text_vocab_size - 1)
        else:                            # VA expert
            labels_adj = (global_labels - self.text_vocab_size).clamp(
                min=0, max=self.va_vocab_size - 1
            )
        per_expert_loss = tensor_parallel.vocab_parallel_cross_entropy(
            local_logits, labels_adj
        )

        # ---- reverse A2A: redistribute losses to original token positions ---
        # combined_loss is [N]: text losses first, VA losses second (permuted).
        combined_loss = self._ep2_combine_loss(
            per_expert_loss, input_splits, output_splits, ep_group
        )

        # Scatter back into the original token order.
        text_count = text_indices.numel()
        per_token_lm_loss = torch.zeros(N, dtype=torch.float32, device=device)
        per_token_lm_loss.index_add_(0, text_indices, combined_loss[:text_count].float())
        per_token_lm_loss.index_add_(0, va_indices, combined_loss[text_count:].float())

        lm_loss_bs = per_token_lm_loss.view(s, b).transpose(0, 1).contiguous()
        result.lm_loss = lm_loss_bs

        router_loss = router_modality_loss(router_logits, modality_target)
        result.router_loss = router_loss

        if loss_mask is not None:
            mask = loss_mask.float()
            lm_scalar = (lm_loss_bs * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            lm_scalar = lm_loss_bs.mean()

        result.total_loss = lm_scalar + self.head_config.router_loss_coeff * router_loss
