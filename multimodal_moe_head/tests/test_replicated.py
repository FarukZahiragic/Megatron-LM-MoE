"""Tests for the replicated (EP=1) path of the modality-routed output head.

How to run (cluster, 2 GPUs)
-----------------------------
    sbatch multimodal_moe_head/tests/run_test_replicated.sh

How to run locally (1 GPU)
---------------------------
    RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29501 \\
        python -m pytest multimodal_moe_head/tests/test_replicated.py -v -s

CUDA is required. The tests use `use_cpu_initialization=True` for deterministic
weight init but call `.cuda()` on all models and tensors before the forward pass.
With world_size=2 (EP=1, DP=2) both ranks run the full test suite independently
and should produce identical results.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.transformer.transformer_config import TransformerConfig

# Ensure the package is importable when running this file directly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from multimodal_moe_head import MoEOutputHead, MoEOutputHeadConfig  # noqa: E402


def _maybe_init_distributed():
    """Initialize a distributed group for the test, reading rank/world_size from environment."""
    if torch.distributed.is_initialized():
        return
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", world_size=world_size, rank=rank)


def _setup_model_parallel():
    _maybe_init_distributed()
    if not parallel_state.is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )


def _make_config(hidden_size: int) -> TransformerConfig:
    return TransformerConfig(
        num_layers=2,
        hidden_size=hidden_size,
        num_attention_heads=4,
        use_cpu_initialization=True,
        params_dtype=torch.float32,
        bf16=False,
        add_bias_linear=False,
    )


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    _setup_model_parallel()
    yield
    if parallel_state.is_initialized():
        parallel_state.destroy_model_parallel()


def _build_head(hidden_size=8, text_vocab=16, total_vocab=32, teacher_force_steps=0):
    config = _make_config(hidden_size)
    head_config = MoEOutputHeadConfig(
        total_vocab_size=total_vocab,
        text_vocab_size=text_vocab,
        teacher_force_steps=teacher_force_steps,
        router_loss_coeff=0.0,
    )
    head = MoEOutputHead(config=config, head_config=head_config).cuda()
    return head, config, head_config


def test_construct_and_shapes():
    head, _config, head_config = _build_head()
    assert head.experts.has_expert(0)
    assert head.experts.has_expert(1)

    # Each projection should hold the full sub-vocab since TP=1.
    text_w = head.experts.projections["0"].weight
    va_w = head.experts.projections["1"].weight
    assert text_w.shape == (head_config.text_vocab_size, 8)
    assert va_w.shape == (head_config.va_vocab_size, 8)


def test_teacher_forced_routing_partitions_tokens_by_label():
    head, _config, head_config = _build_head(teacher_force_steps=10)
    s, b, h = 4, 2, 8
    torch.manual_seed(0)
    hidden = torch.randn(s, b, h).cuda()

    # Construct labels with a known split.
    T = head_config.text_vocab_size
    labels = torch.tensor(
        [[0,     T    ],
         [T + 1, 1    ],
         [2,     3    ],
         [T + 2, T + 3]],
    ).T.cuda()  # transpose so shape is [b, s]

    result = head(hidden, labels=labels, current_step=0)

    # Under teacher forcing, routing must agree with ground truth modality.
    flat_labels = labels.transpose(0, 1).reshape(-1)
    expected_modality = (flat_labels >= head_config.text_vocab_size).long()
    assert torch.equal(result.routing_decision, expected_modality)

    # Each expert should have processed exactly the tokens of its modality.
    text_count = (expected_modality == 0).sum().item()
    va_count = (expected_modality == 1).sum().item()
    assert result.text_token_indices.numel() == text_count
    assert result.va_token_indices.numel() == va_count


def test_loss_matches_hand_rolled_reference_under_teacher_forcing():
    """LM cross-entropy through the head must equal a manual reference computation."""
    head, _config, head_config = _build_head(teacher_force_steps=10)
    s, b, h = 4, 2, 8
    torch.manual_seed(42)
    hidden = torch.randn(s, b, h).cuda()
    T = head_config.text_vocab_size
    labels = torch.tensor(
        [[0,     T,     1,     T + 1],
         [T + 2, 2,     T + 3, 3   ]],
    ).cuda()

    result = head(hidden, labels=labels, current_step=0)

    # Hand-rolled reference: directly apply each expert's weight to the full
    # hidden tensor, take cross-entropy on the matching positions.
    text_weight = head.experts.projections["0"].weight
    va_weight = head.experts.projections["1"].weight

    flat_hidden = hidden.reshape(s * b, h)
    flat_labels = labels.transpose(0, 1).reshape(-1)
    expected_modality = (flat_labels >= head_config.text_vocab_size).long()

    text_mask = expected_modality == 0
    va_mask = expected_modality == 1

    text_logits_ref = flat_hidden[text_mask] @ text_weight.T
    va_logits_ref = flat_hidden[va_mask] @ va_weight.T

    text_loss_ref = F.cross_entropy(
        text_logits_ref, flat_labels[text_mask], reduction="none"
    )
    va_loss_ref = F.cross_entropy(
        va_logits_ref, flat_labels[va_mask] - head_config.text_vocab_size, reduction="none"
    )

    ref_per_token = torch.zeros(s * b, dtype=torch.float32, device=hidden.device)
    ref_per_token[text_mask] = text_loss_ref
    ref_per_token[va_mask] = va_loss_ref
    ref_loss_bs = ref_per_token.view(s, b).transpose(0, 1).contiguous()

    assert torch.allclose(result.lm_loss, ref_loss_bs, atol=1e-4, rtol=1e-4), (
        f"head lm_loss differs from hand-rolled reference\n"
        f"  head: {result.lm_loss}\n  ref:  {ref_loss_bs}"
    )


def test_router_loss_matches_manual_cross_entropy():
    head, _config, head_config = _build_head(teacher_force_steps=10)
    head.head_config.router_loss_coeff = 1.0
    s, b, h = 4, 2, 8
    torch.manual_seed(7)
    hidden = torch.randn(s, b, h).cuda()
    labels = torch.tensor(
        [[3, head_config.text_vocab_size + 1, 2, head_config.text_vocab_size + 7],
         [head_config.text_vocab_size + 5, 7, head_config.text_vocab_size + 2, 4]],
    ).cuda()
    result = head(hidden, labels=labels, current_step=0)

    flat_hidden = hidden.reshape(s * b, h)
    flat_labels = labels.transpose(0, 1).reshape(-1)
    expected_modality = (flat_labels >= head_config.text_vocab_size).long()

    expected_router_logits = head.router(flat_hidden.float())
    expected_router_loss = F.cross_entropy(expected_router_logits, expected_modality)

    assert torch.allclose(result.router_loss, expected_router_loss, atol=1e-5)


def test_inference_path_argmax_routing():
    head, _config, _hc = _build_head()
    s, b, h = 3, 2, 8
    torch.manual_seed(1)
    hidden = torch.randn(s, b, h).cuda()
    result = head(hidden, labels=None)

    expected_choice = result.router_logits.argmax(dim=-1)
    assert torch.equal(result.routing_decision, expected_choice)
    assert result.lm_loss is None
    assert result.router_loss is None
    assert result.total_loss is None


def test_backward_gradients_are_populated():
    """total_loss.backward() populates grads on router and both expert projections."""
    head, _config, head_config = _build_head(teacher_force_steps=10)
    head.head_config.router_loss_coeff = 1.0   # router CE enters the loss so router gets a grad
    s, b, h = 4, 2, 8
    torch.manual_seed(42)
    hidden = torch.randn(s, b, h, device='cuda', requires_grad=True)  # leaf on CUDA
    T = head_config.text_vocab_size
    labels = torch.tensor(
        [[0,     T,     1,     T + 1],
         [T + 2, 2,     T + 3, 3   ]],
    ).cuda()  # mixed modalities — both experts used

    result = head(hidden, labels=labels, current_step=0)
    assert result.total_loss is not None
    result.total_loss.backward()

    assert head.router.weight.grad is not None, "router weight has no grad"
    assert head.router.weight.grad.abs().sum() > 0, "router weight grad is all-zero"

    w0 = head.experts.projections["0"].weight
    w1 = head.experts.projections["1"].weight
    assert w0.grad is not None, "text expert weight has no grad"
    assert w1.grad is not None, "VA expert weight has no grad"
    assert w0.grad.abs().sum() > 0, "text expert weight grad is all-zero"
    assert w1.grad.abs().sum() > 0, "VA expert weight grad is all-zero"

    assert hidden.grad is not None, "hidden has no grad"
    assert hidden.grad.shape == hidden.shape


def test_backward_unused_expert_has_zero_grad():
    """An unused expert must get an all-ZERO grad (not None) when it gets no tokens.

    The head always runs both expert forwards (on a [0, H] input -> [0, V] when
    an expert receives no tokens), so the unused expert's weight is always in the
    autograd graph and gets a deterministic zero grad. This is what keeps the
    gradient-reduction collectives in lockstep across data-parallel ranks under
    --overlap-grad-reduce (a routing-dependent grad=None desynchronizes them and
    deadlocks NCCL).
    """
    head, _config, head_config = _build_head(teacher_force_steps=10)
    s, b, h = 4, 2, 8
    torch.manual_seed(3)
    hidden = torch.randn(s, b, h, device='cuda', requires_grad=True)  # leaf on CUDA
    # All labels in text range → teacher forcing routes every token to expert 0.
    labels = torch.tensor(
        [[0, 1, 2, 3],
         [4, 5, 6, 7]],
    ).cuda()

    result = head(hidden, labels=labels, current_step=0)
    result.total_loss.backward()

    w0 = head.experts.projections["0"].weight
    w1 = head.experts.projections["1"].weight
    assert w0.grad is not None and w0.grad.abs().sum() > 0, "text expert must have nonzero grad"
    assert w1.grad is not None and w1.grad.abs().sum() == 0, (
        "VA expert must have an all-zero (not None) grad when it processed no tokens"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-vv", "-s"]))
