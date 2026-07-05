"""Tests for the EP=2 (one expert per rank) path of the modality-routed output head.

Topology
--------
world_size=4, TP=1, PP=1, EP=2, DP=2.

    rank 0 — EP rank 0 (text expert, DP replica 0)
    rank 1 — EP rank 1 (VA expert,   DP replica 0)
    rank 2 — EP rank 0 (text expert, DP replica 1)
    rank 3 — EP rank 1 (VA expert,   DP replica 1)

EP groups: [0,1] and [2,3].
DP groups: [0,2] and [1,3].

All four ranks run the full test suite. Within each EP group, both ranks
see the same micro-batch (same hidden states and labels) and should produce
identical loss/loss-gradient values.

The tests also run with world_size=2 (EP=2, DP=1) to allow local execution
with only 2 GPUs.

How to run (cluster, 4 GPUs)
-----------------------------
    sbatch multimodal_moe_head/tests/run_test_ep2.sh

How to run locally (2 GPUs, EP=2, DP=1)
-----------------------------------------
    RANK=0 LOCAL_RANK=0 WORLD_SIZE=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=29502 \\
        python -m pytest multimodal_moe_head/tests/test_ep2.py -v -s &
    RANK=1 LOCAL_RANK=1 WORLD_SIZE=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=29502 \\
        python -m pytest multimodal_moe_head/tests/test_ep2.py -v -s

CUDA is required.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.transformer.transformer_config import TransformerConfig

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from multimodal_moe_head import MoEOutputHead, MoEOutputHeadConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Distributed / parallel-state setup
# ---------------------------------------------------------------------------

def _maybe_init_distributed() -> None:
    if dist.is_initialized():
        return
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "2"))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29502")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)


def _setup_model_parallel() -> None:
    _maybe_init_distributed()
    if not parallel_state.is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=2,
        )


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    _setup_model_parallel()
    yield
    if parallel_state.is_initialized():
        parallel_state.destroy_model_parallel()


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

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


def _build_head(
    hidden_size: int = 8,
    text_vocab: int = 16,
    total_vocab: int = 32,
    teacher_force_steps: int = 0,
    router_loss_coeff: float = 0.0,
) -> tuple[MoEOutputHead, TransformerConfig, MoEOutputHeadConfig]:
    config = _make_config(hidden_size)
    head_config = MoEOutputHeadConfig(
        total_vocab_size=total_vocab,
        text_vocab_size=text_vocab,
        teacher_force_steps=teacher_force_steps,
        router_loss_coeff=router_loss_coeff,
        expert_parallel_size=2,
    )
    head = MoEOutputHead(config=config, head_config=head_config).cuda()
    return head, config, head_config


def _ep_rank() -> int:
    return parallel_state.get_expert_model_parallel_rank()


def _ep_group() -> dist.ProcessGroup:
    return parallel_state.get_expert_model_parallel_group()



def _all_ranks_equal(t: torch.Tensor) -> bool:
    """Return True if all ranks in the EP group have the same tensor values."""
    ref = t.clone()
    dist.broadcast(ref, src=dist.get_global_rank(_ep_group(), 0), group=_ep_group())
    return bool(torch.allclose(t, ref, atol=0.0, rtol=0.0))


# ---------------------------------------------------------------------------
# Test 1: construction and weight shapes
# ---------------------------------------------------------------------------

def test_construct_and_shapes():
    head, _config, head_config = _build_head()
    ep_rank = _ep_rank()

    if ep_rank == 0:
        # Text expert lives on rank 0 only.
        assert head.experts.has_expert(0), "EP rank 0 must hold text expert"
        assert not head.experts.has_expert(1), "EP rank 0 must NOT hold VA expert"
        w = head.experts.projections["0"].weight
        assert w.shape == (head_config.text_vocab_size, 8), (
            f"Expected ({head_config.text_vocab_size}, 8), got {w.shape}"
        )
    else:
        # VA expert lives on rank 1 only.
        assert not head.experts.has_expert(0), "EP rank 1 must NOT hold text expert"
        assert head.experts.has_expert(1), "EP rank 1 must hold VA expert"
        w = head.experts.projections["1"].weight
        assert w.shape == (head_config.va_vocab_size, 8), (
            f"Expected ({head_config.va_vocab_size}, 8), got {w.shape}"
        )


# ---------------------------------------------------------------------------
# Test 2: teacher-forced routing assigns tokens to correct expert rank
# ---------------------------------------------------------------------------

def test_teacher_forced_routing_partitions_tokens():
    head, _config, head_config = _build_head(teacher_force_steps=10)
    s, b, h = 4, 2, 8
    torch.manual_seed(0)
    hidden = torch.randn(s, b, h, device="cuda")

    T = head_config.text_vocab_size
    labels = torch.tensor(
        [[0,     T    ],
         [T + 1, 1    ],
         [2,     T + 2],
         [T + 3, 3    ]],
    ).T.cuda()  # [b=2, s=4]

    result = head(hidden, labels=labels, current_step=0)

    flat_labels = labels.transpose(0, 1).reshape(-1)
    expected_modality = (flat_labels >= head_config.text_vocab_size).long()
    assert torch.equal(result.routing_decision, expected_modality), (
        f"routing_decision mismatch on EP rank {_ep_rank()}"
    )


# ---------------------------------------------------------------------------
# Test 3: after dispatch EP rank 0 sees only text tokens, rank 1 only VA
# ---------------------------------------------------------------------------

def test_dispatch_gives_correct_tokens_to_each_expert():
    """Verify which tokens each expert rank processes by inspecting logit shapes."""
    head, _config, head_config = _build_head(teacher_force_steps=10)
    s, b, h = 4, 2, 8
    torch.manual_seed(5)
    hidden = torch.randn(s, b, h, device="cuda")
    N = s * b

    T = head_config.text_vocab_size
    labels = torch.tensor(
        [[0,     T,     1,     T + 1],
         [T + 2, 2,     T + 3, 3   ]],
    ).cuda()  # [b=2, s=4]

    result = head(hidden, labels=labels, current_step=0)

    flat_labels = labels.transpose(0, 1).reshape(-1)
    text_count = (flat_labels < head_config.text_vocab_size).sum().item()
    va_count = N - text_count

    ep_rank = _ep_rank()
    if ep_rank == 0:
        assert result.text_logits is not None, "EP rank 0 must produce text_logits"
        assert result.va_logits is None, "EP rank 0 must not produce va_logits"
        assert result.text_logits.shape[0] == text_count * dist.get_world_size(_ep_group()), (
            f"EP rank 0 expected {text_count * dist.get_world_size(_ep_group())} text tokens "
            f"(one copy per DP rank in this EP group), got {result.text_logits.shape[0]}"
        )
    else:
        assert result.va_logits is not None, "EP rank 1 must produce va_logits"
        assert result.text_logits is None, "EP rank 1 must not produce text_logits"
        assert result.va_logits.shape[0] == va_count * dist.get_world_size(_ep_group()), (
            f"EP rank 1 expected {va_count * dist.get_world_size(_ep_group())} VA tokens, "
            f"got {result.va_logits.shape[0]}"
        )


# ---------------------------------------------------------------------------
# Test 4: both EP ranks reconstruct identical lm_loss
# ---------------------------------------------------------------------------

def test_both_ep_ranks_get_identical_lm_loss():
    head, _config, head_config = _build_head(teacher_force_steps=10)
    s, b, h = 4, 2, 8
    torch.manual_seed(42)
    hidden = torch.randn(s, b, h, device="cuda")
    T = head_config.text_vocab_size
    labels = torch.tensor(
        [[0,     T,     1,     T + 1],
         [T + 2, 2,     T + 3, 3   ]],
    ).cuda()

    result = head(hidden, labels=labels, current_step=0)

    assert result.lm_loss is not None
    assert result.total_loss is not None

    # Both ranks in the EP group must agree on every per-token loss.
    assert _all_ranks_equal(result.lm_loss), (
        f"EP ranks disagree on lm_loss (rank {_ep_rank()}): {result.lm_loss}"
    )
    assert _all_ranks_equal(result.total_loss.unsqueeze(0)), (
        f"EP ranks disagree on total_loss (rank {_ep_rank()}): {result.total_loss}"
    )


# ---------------------------------------------------------------------------
# Test 5: LM loss matches hand-rolled reference under teacher forcing
# ---------------------------------------------------------------------------

def test_lm_loss_matches_hand_rolled_reference():
    """EP=2 LM loss must equal a direct cross-entropy reference on each rank."""
    head, _config, head_config = _build_head(teacher_force_steps=10)
    s, b, h = 4, 2, 8
    torch.manual_seed(42)
    hidden = torch.randn(s, b, h, device="cuda")
    T = head_config.text_vocab_size
    labels = torch.tensor(
        [[0,     T,     1,     T + 1],
         [T + 2, 2,     T + 3, 3   ]],
    ).cuda()

    result = head(hidden, labels=labels, current_step=0)

    flat_hidden = hidden.reshape(s * b, h)
    flat_labels = labels.transpose(0, 1).reshape(-1)
    text_mask = flat_labels < head_config.text_vocab_size
    va_mask = ~text_mask
    text_indices = text_mask.nonzero(as_tuple=False).squeeze(-1)
    va_indices = va_mask.nonzero(as_tuple=False).squeeze(-1)

    ep_rank = _ep_rank()
    # Boolean masks in [b, s] space — same shape as result.lm_loss.
    text_mask_bs = labels < head_config.text_vocab_size   # [b, s]
    va_mask_bs = ~text_mask_bs

    if ep_rank == 0:
        # Build reference using *this rank's* text projection weight.
        text_weight = head.experts.projections["0"].weight
        # The expert received 2*text_count tokens (one copy per DP rank).
        # Each copy is the same, so the reference uses just the local tokens.
        text_logits_ref = flat_hidden[text_indices] @ text_weight.T
        text_loss_ref = F.cross_entropy(
            text_logits_ref, flat_labels[text_indices], reduction="none"
        )
        ref_per_token = torch.zeros(s * b, dtype=torch.float32, device="cuda")
        ref_per_token[text_indices] = text_loss_ref
        # ref_per_token is in flat (s,b) order; reshape to [s,b] then T → [b,s]
        ref_lm_loss = ref_per_token.view(s, b).T.contiguous()
        # Text positions must match the reference.
        if text_mask_bs.any():
            assert torch.allclose(
                result.lm_loss[text_mask_bs],
                ref_lm_loss[text_mask_bs],
                atol=1e-4, rtol=1e-4,
            ), (
                f"EP rank 0: text token LM losses differ from hand-rolled reference\n"
                f"  head:  {result.lm_loss[text_mask_bs]}\n"
                f"  ref:   {ref_lm_loss[text_mask_bs]}"
            )
        # VA positions come from rank 1 via combine A2A — must be nonzero.
        if va_mask_bs.any():
            assert result.lm_loss[va_mask_bs].abs().sum() > 0, (
                "EP rank 0: VA positions in lm_loss are all zero — combine A2A may have failed"
            )
    else:
        va_weight = head.experts.projections["1"].weight
        va_logits_ref = flat_hidden[va_indices] @ va_weight.T
        va_loss_ref = F.cross_entropy(
            va_logits_ref,
            flat_labels[va_indices] - head_config.text_vocab_size,
            reduction="none",
        )
        ref_per_token = torch.zeros(s * b, dtype=torch.float32, device="cuda")
        ref_per_token[va_indices] = va_loss_ref
        ref_lm_loss = ref_per_token.view(s, b).T.contiguous()
        # VA positions must match the reference.
        if va_mask_bs.any():
            assert torch.allclose(
                result.lm_loss[va_mask_bs],
                ref_lm_loss[va_mask_bs],
                atol=1e-4, rtol=1e-4,
            ), (
                f"EP rank 1: VA token LM losses differ from hand-rolled reference\n"
                f"  head:  {result.lm_loss[va_mask_bs]}\n"
                f"  ref:   {ref_lm_loss[va_mask_bs]}"
            )
        # Text positions come from rank 0 — must be nonzero.
        if text_mask_bs.any():
            assert result.lm_loss[text_mask_bs].abs().sum() > 0, (
                "EP rank 1: text positions in lm_loss are all zero — combine A2A may have failed"
            )


# ---------------------------------------------------------------------------
# Test 6: router loss matches manual cross-entropy
# ---------------------------------------------------------------------------

def test_router_loss_matches_manual():
    head, _config, head_config = _build_head(teacher_force_steps=10, router_loss_coeff=1.0)
    s, b, h = 4, 2, 8
    torch.manual_seed(7)
    hidden = torch.randn(s, b, h, device="cuda")
    T = head_config.text_vocab_size
    labels = torch.tensor(
        [[0,     T,     1,     T + 1],
         [T + 2, 2,     T + 3, 3   ]],
    ).cuda()

    result = head(hidden, labels=labels, current_step=0)

    flat_hidden = hidden.reshape(s * b, h)
    flat_labels = labels.transpose(0, 1).reshape(-1)
    modality = (flat_labels >= head_config.text_vocab_size).long()
    expected = F.cross_entropy(head.router(flat_hidden.float()), modality)

    assert torch.allclose(result.router_loss, expected, atol=1e-5), (
        f"router_loss mismatch on EP rank {_ep_rank()}: "
        f"{result.router_loss} vs {expected}"
    )


# ---------------------------------------------------------------------------
# Test 7: inference path — routing_decision follows argmax(router_logits)
# ---------------------------------------------------------------------------

def test_inference_path_argmax_routing():
    head, _config, _hc = _build_head()
    s, b, h = 3, 2, 8
    torch.manual_seed(1)
    hidden = torch.randn(s, b, h, device="cuda")
    result = head(hidden, labels=None)

    expected_choice = result.router_logits.argmax(dim=-1)
    assert torch.equal(result.routing_decision, expected_choice), (
        f"Inference routing_decision != argmax on EP rank {_ep_rank()}"
    )
    assert result.lm_loss is None
    assert result.router_loss is None
    assert result.total_loss is None


# ---------------------------------------------------------------------------
# Test 8: all-text labels — EP rank 1 gets no tokens; grad is all-zero (not None)
# ---------------------------------------------------------------------------

def test_all_text_labels():
    head, _config, _ = _build_head(teacher_force_steps=10, router_loss_coeff=1.0)
    s, b, h = 4, 2, 8
    torch.manual_seed(3)
    hidden = torch.randn(s, b, h, device="cuda", requires_grad=True)
    # All labels are in the text range.
    labels = torch.tensor(
        [[0, 1, 2, 3],
         [4, 5, 6, 7]],
    ).cuda()

    result = head(hidden, labels=labels, current_step=0)
    assert result.total_loss is not None
    result.total_loss.backward()

    ep_rank = _ep_rank()
    w0 = head.experts.projections["0"] if "0" in head.experts.projections else None
    w1 = head.experts.projections["1"] if "1" in head.experts.projections else None

    if ep_rank == 0:
        assert w0 is not None
        assert w0.weight.grad is not None and w0.weight.grad.abs().sum() > 0, (
            "EP rank 0 text expert must have nonzero grad for all-text batch"
        )
    else:
        assert w1 is not None
        # The VA expert always runs (on a [0, H] input -> [0, V]) so its weight
        # gets an all-ZERO grad, not None. This is intentional: a deterministic
        # grad keeps the gradient-reduction collectives in lockstep across DP
        # replicas under --overlap-grad-reduce (see head.py forward note).
        assert w1.weight.grad is not None and w1.weight.grad.abs().sum() == 0, (
            "EP rank 1 VA expert must have an all-zero (not None) grad when it "
            "processed no VA tokens"
        )

    assert hidden.grad is not None
    assert hidden.grad.shape == hidden.shape


# ---------------------------------------------------------------------------
# Test 9: all-VA labels — EP rank 0 gets no tokens; grad is all-zero (not None)
# ---------------------------------------------------------------------------

def test_all_va_labels():
    head, _config, head_config = _build_head(teacher_force_steps=10, router_loss_coeff=1.0)
    s, b, h = 4, 2, 8
    torch.manual_seed(4)
    hidden = torch.randn(s, b, h, device="cuda", requires_grad=True)
    # All labels are in the VA range.
    T = head_config.text_vocab_size
    labels = torch.tensor(
        [[T,     T + 1, T + 2, T + 3],
         [T + 4, T + 5, T + 6, T + 7]],
    ).cuda()

    result = head(hidden, labels=labels, current_step=0)
    assert result.total_loss is not None
    result.total_loss.backward()

    ep_rank = _ep_rank()
    w0 = head.experts.projections["0"] if "0" in head.experts.projections else None
    w1 = head.experts.projections["1"] if "1" in head.experts.projections else None

    if ep_rank == 0:
        assert w0 is not None
        # Text expert always runs (on [0, H]) -> all-zero grad, not None. See
        # the note in test_all_text_labels / head.py.
        assert w0.weight.grad is not None and w0.weight.grad.abs().sum() == 0, (
            "EP rank 0 text expert must have an all-zero (not None) grad when it "
            "processed no text tokens"
        )
    else:
        assert w1 is not None
        assert w1.weight.grad is not None and w1.weight.grad.abs().sum() > 0, (
            "EP rank 1 VA expert must have nonzero grad for all-VA batch"
        )

    assert hidden.grad is not None
    assert hidden.grad.shape == hidden.shape


# ---------------------------------------------------------------------------
# Test 10: mixed labels — both experts used, both grads populated
# ---------------------------------------------------------------------------

def test_mixed_labels_backward_grads_populated():
    head, _config, head_config = _build_head(teacher_force_steps=10, router_loss_coeff=1.0)
    s, b, h = 4, 2, 8
    torch.manual_seed(42)
    hidden = torch.randn(s, b, h, device="cuda", requires_grad=True)
    T = head_config.text_vocab_size
    labels = torch.tensor(
        [[0,     T,     1,     T + 1],
         [T + 2, 2,     T + 3, 3   ]],
    ).cuda()

    result = head(hidden, labels=labels, current_step=0)
    assert result.total_loss is not None
    result.total_loss.backward()

    ep_rank = _ep_rank()

    # Router is replicated — must always have a nonzero grad.
    assert head.router.weight.grad is not None, "router weight has no grad"
    assert head.router.weight.grad.abs().sum() > 0, "router weight grad is all-zero"

    # Local expert must have a nonzero grad.
    local_key = "0" if ep_rank == 0 else "1"
    w_local = head.experts.projections[local_key]
    assert w_local.weight.grad is not None, f"EP rank {ep_rank} local expert has no grad"
    assert w_local.weight.grad.abs().sum() > 0, f"EP rank {ep_rank} local expert grad is all-zero"

    # hidden must have grad flowing from both experts (via combine A2A backward).
    assert hidden.grad is not None, "hidden has no grad"
    assert hidden.grad.shape == hidden.shape


# ---------------------------------------------------------------------------
# Test 11: gradient on hidden flows back through A2A
# ---------------------------------------------------------------------------

def test_hidden_grad_shape_and_nonzero():
    """hidden.grad must be non-zero — the expert grad flows back via combine A2A."""
    head, _config, head_config = _build_head(teacher_force_steps=10)
    s, b, h = 4, 2, 8
    torch.manual_seed(9)
    hidden = torch.randn(s, b, h, device="cuda", requires_grad=True)
    T = head_config.text_vocab_size
    labels = torch.tensor(
        [[0,     T,     1,     T + 1],
         [T + 2, 2,     T + 3, 3   ]],
    ).cuda()

    result = head(hidden, labels=labels, current_step=0)
    result.total_loss.backward()

    assert hidden.grad is not None
    assert hidden.grad.shape == hidden.shape
    assert hidden.grad.abs().sum() > 0, "hidden.grad is all-zero — backward may have broken"


# ---------------------------------------------------------------------------
# Test 12: argmax routing (no teacher forcing) — routing matches router output
# ---------------------------------------------------------------------------

def test_free_routing_follows_argmax():
    head, _config, head_config = _build_head(teacher_force_steps=0)
    s, b, h = 4, 2, 8
    torch.manual_seed(13)
    hidden = torch.randn(s, b, h, device="cuda")
    T = head_config.text_vocab_size
    labels = torch.tensor(
        [[0,     T,     1,     T + 1],
         [T + 2, 2,     T + 3, 3   ]],
    ).cuda()

    result = head(hidden, labels=labels, current_step=100)  # beyond teacher_force_steps=0
    expected = result.router_logits.argmax(dim=-1)
    assert torch.equal(result.routing_decision, expected), (
        f"Free routing_decision != argmax on EP rank {_ep_rank()}"
    )


# ---------------------------------------------------------------------------
# Test 14: total_loss is finite and positive
# ---------------------------------------------------------------------------

def test_total_loss_is_finite():
    head, _config, head_config = _build_head(teacher_force_steps=10, router_loss_coeff=0.01)
    s, b, h = 4, 2, 8
    torch.manual_seed(14)
    hidden = torch.randn(s, b, h, device="cuda")
    T = head_config.text_vocab_size
    labels = torch.tensor(
        [[0,     T,     1,     T + 1],
         [T + 2, 2,     T + 3, 3   ]],
    ).cuda()

    result = head(hidden, labels=labels, current_step=0)

    assert result.total_loss is not None
    assert torch.isfinite(result.total_loss), f"total_loss is not finite: {result.total_loss}"
    assert result.total_loss.item() > 0, f"total_loss is non-positive: {result.total_loss}"


# ---------------------------------------------------------------------------
# Test 15: lm_loss shape is [b, s]
# ---------------------------------------------------------------------------

def test_lm_loss_shape():
    s, b, h = 6, 3, 8
    head, _config, head_config = _build_head(hidden_size=h, teacher_force_steps=10)
    torch.manual_seed(15)
    hidden = torch.randn(s, b, h, device="cuda")
    T = head_config.text_vocab_size
    labels = torch.zeros(b, s, dtype=torch.long, device="cuda")
    labels[0, 0] = T
    labels[1, 2] = T
    labels[2, 4] = T

    result = head(hidden, labels=labels, current_step=0)
    assert result.lm_loss is not None
    assert result.lm_loss.shape == (b, s), (
        f"Expected lm_loss shape ({b}, {s}), got {result.lm_loss.shape}"
    )


# ---------------------------------------------------------------------------
# Test 16: EP=2 numerics match the EP=1 hand-rolled reference (full loss)
# ---------------------------------------------------------------------------

def test_ep2_full_loss_matches_ep1_reference():
    """The complete lm_loss must equal the hand-rolled EP=1 reference.

    Each rank computes the reference for its own expert's tokens using the
    local weight, then both checks are verified independently.
    After the combine A2A, both ranks hold the full [b,s] lm_loss. We
    compare against a local partial reference (only the positions for which
    this rank owns the expert), then check that off-modality positions are
    populated via the combine A2A.
    """
    head, _config, head_config = _build_head(teacher_force_steps=10)
    s, b, h = 4, 2, 8
    torch.manual_seed(2025)
    hidden = torch.randn(s, b, h, device="cuda")
    T = head_config.text_vocab_size
    labels = torch.tensor(
        [[0,     T,     1,     T + 1],
         [T + 2, 2,     T + 3, 3   ]],
    ).cuda()

    result = head(hidden, labels=labels, current_step=0)

    flat_hidden = hidden.reshape(s * b, h)
    flat_labels = labels.transpose(0, 1).reshape(-1)
    text_mask = flat_labels < head_config.text_vocab_size
    va_mask = ~text_mask
    text_indices = text_mask.nonzero(as_tuple=False).squeeze(-1)
    va_indices = va_mask.nonzero(as_tuple=False).squeeze(-1)

    ep_rank = _ep_rank()

    # Reconstruct what a single EP rank's expert contributes.
    ref_per_token = torch.zeros(s * b, dtype=torch.float32, device="cuda")

    if ep_rank == 0:
        text_w = head.experts.projections["0"].weight
        tl = flat_hidden[text_indices] @ text_w.T
        ref_per_token[text_indices] = F.cross_entropy(
            tl, flat_labels[text_indices], reduction="none"
        )
        # VA positions come from rank 1 — verify they're nonzero
        # (not zero-filled) after combine.
        if va_indices.numel() > 0:
            assert result.lm_loss.view(-1)[va_indices].abs().sum() > 0, (
                "EP rank 0: VA positions in lm_loss are all zero — combine A2A may have failed"
            )
    else:
        va_w = head.experts.projections["1"].weight
        vl = flat_hidden[va_indices] @ va_w.T
        ref_per_token[va_indices] = F.cross_entropy(
            vl, flat_labels[va_indices] - head_config.text_vocab_size, reduction="none"
        )
        # Text positions come from rank 0 — verify they're nonzero.
        if text_indices.numel() > 0:
            assert result.lm_loss.view(-1)[text_indices].abs().sum() > 0, (
                "EP rank 1: text positions in lm_loss are all zero — combine A2A may have failed"
            )

    # Check the local expert's contribution exactly.
    # ref_per_token is in flat [s,b] order; reshape to [b,s] to compare against lm_loss.
    ref_lm_loss = ref_per_token.view(s, b).T.contiguous()
    own_mask_bs = (labels < head_config.text_vocab_size) if ep_rank == 0 else ~(labels < head_config.text_vocab_size)
    if own_mask_bs.any():
        assert torch.allclose(
            result.lm_loss[own_mask_bs],
            ref_lm_loss[own_mask_bs],
            atol=1e-4, rtol=1e-4,
        ), (
            f"EP rank {ep_rank}: local expert loss positions mismatch.\n"
            f"  head: {result.lm_loss[own_mask_bs]}\n"
            f"  ref:  {ref_lm_loss[own_mask_bs]}"
        )


# ---------------------------------------------------------------------------
# Test 17: loss_mask propagates correctly
# ---------------------------------------------------------------------------

def test_loss_mask_applied():
    head, _config, head_config = _build_head(teacher_force_steps=10)
    s, b, h = 4, 2, 8
    torch.manual_seed(17)
    hidden = torch.randn(s, b, h, device="cuda")
    T = head_config.text_vocab_size
    labels = torch.tensor(
        [[0,     T,     1,     T + 1],
         [T + 2, 2,     T + 3, 3   ]],
    ).cuda()

    # Mask out every token.
    zero_mask = torch.zeros(b, s, device="cuda")
    result_masked = head(hidden, labels=labels, current_step=0, loss_mask=zero_mask)

    # With all-zero mask, lm_scalar = 0 and total_loss = router + lb contributions.
    assert result_masked.total_loss is not None
    # lm contribution is 0 (or close to it given router/lb coefficients are 0).
    assert result_masked.total_loss.item() < 1e-6, (
        f"Expected near-zero total_loss with all-zero mask, got {result_masked.total_loss}"
    )


# ---------------------------------------------------------------------------
# Test 18: DP replicas receive and process independent micro-batches
# ---------------------------------------------------------------------------

def test_dp_replicas_process_independent_batches():
    """DP replicas must process independent micro-batches and produce independent outputs.

    Each EP group seeds its hidden states from its DP rank, so EP group 0
    (ranks 0,1) and EP group 1 (ranks 2,3) get different inputs. With
    world_size=4 (DP=2), total_loss must differ between EP groups.
    With world_size=2 (DP=1) there is only one EP group so the check is skipped.
    """
    dp_rank = parallel_state.get_data_parallel_rank()
    dp_size = parallel_state.get_data_parallel_world_size()
    dp_group = parallel_state.get_data_parallel_group()

    head, _config, head_config = _build_head(teacher_force_steps=10)
    s, b, h = 4, 2, 8

    # Seed differs per DP replica so EP groups get genuinely different inputs.
    torch.manual_seed(dp_rank)
    hidden = torch.randn(s, b, h, device="cuda")
    T = head_config.text_vocab_size
    labels = torch.tensor(
        [[0,     T,     1,     T + 1],
         [T + 2, 2,     T + 3, 3   ]],
    ).cuda()

    result = head(hidden, labels=labels, current_step=0)

    if dp_size > 1:
        # Across DP replicas, losses must differ (different inputs → different outputs).
        loss_val = result.total_loss.detach().clone()
        all_losses = [torch.zeros_like(loss_val) for _ in range(dp_size)]
        dist.all_gather(all_losses, loss_val, group=dp_group)
        assert not torch.allclose(all_losses[0], all_losses[1], atol=1e-6), (
            "DP replicas produced identical total_loss despite different hidden inputs — "
            "EP groups are not data-parallel-independent."
        )


# ---------------------------------------------------------------------------
# Test 19: DP replica backward passes are independent (no cross-group gradient leakage)
# ---------------------------------------------------------------------------

def test_dp_replicas_backward_independent():
    """Expert weight gradients must differ across DP replicas that saw different batches.

    After backward (no optimizer step, no gradient all-reduce), rank 0 and
    rank 2 both hold the text expert but computed it with different micro-batches.
    Their gradients must be different.  Within the same EP group, both ranks
    must agree on their expert's gradient.

    Skipped when world_size=2 (DP=1) — there is only one replica.
    """
    dp_rank = parallel_state.get_data_parallel_rank()
    dp_size = parallel_state.get_data_parallel_world_size()
    dp_group = parallel_state.get_data_parallel_group()

    if dp_size < 2:
        pytest.skip("DP independence test requires at least 2 DP replicas (world_size >= 4)")

    head, _config, head_config = _build_head(
        teacher_force_steps=10, router_loss_coeff=1.0
    )
    s, b, h = 4, 2, 8

    torch.manual_seed(dp_rank)
    hidden = torch.randn(s, b, h, requires_grad=True, device="cuda")
    T = head_config.text_vocab_size
    labels = torch.tensor(
        [[0,     T,     1,     T + 1],
         [T + 2, 2,     T + 3, 3   ]],
    ).cuda()

    result = head(hidden, labels=labels, current_step=0)
    result.total_loss.backward()

    # Each EP rank owns one expert; get the gradient for that expert's projection.
    ep_rank = _ep_rank()
    expert_key = str(ep_rank)
    proj_weight = head.experts.projections[expert_key]
    assert proj_weight.weight.grad is not None, (
        f"EP rank {ep_rank}: expert projection has no gradient after backward"
    )

    # Across DP replicas (same expert key, different micro-batches), grads must differ.
    grad_flat = proj_weight.weight.grad.detach().reshape(-1)
    all_grads = [torch.zeros_like(grad_flat) for _ in range(dp_size)]
    dist.all_gather(all_grads, grad_flat, group=dp_group)
    assert not torch.allclose(all_grads[0], all_grads[1], atol=1e-6), (
        f"EP rank {ep_rank}: expert weight gradients are identical across DP replicas "
        "despite different micro-batches — EP groups may share state."
    )


# ---------------------------------------------------------------------------
# Test 20: all 4 ranks carry genuinely independent micro-batches (DP=4 for
#           non-expert layers, DP=2 per expert)
# ---------------------------------------------------------------------------

def test_all_ranks_independent_microbatches():
    """With world_size=4, EP=2: non-expert params have DP=4, each expert has DP=2.

    All 4 ranks carry unique micro-batches (seeded by global rank), including
    ranks within the same EP group.  The dispatch A2A routes this rank's text
    tokens to the text expert and VA tokens to the VA expert; the combine A2A
    returns the computed losses back to this rank.

    Verifies routing and loss-return correctness:
    - Each rank's lm_loss at its own-expert positions matches a hand-rolled
      reference computed from *this rank's* hidden states and expert weight.
    - The opposite-modality positions are nonzero, confirming the combine A2A
      returned losses from the other expert for *this rank's* tokens.
    - All 4 ranks produce distinct total_loss values (fully independent).
    - Backward succeeds with nonzero finite expert gradients on every rank.
    """
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    ep_rank = _ep_rank()

    head, _config, head_config = _build_head(teacher_force_steps=10, router_loss_coeff=0.0)
    s, b, h = 4, 2, 8

    # Every rank gets a unique seed — including ranks within the same EP group.
    torch.manual_seed(global_rank)
    hidden = torch.randn(s, b, h, requires_grad=True, device="cuda")
    T = head_config.text_vocab_size
    labels = torch.tensor(
        [[0,     T,     1,     T + 1],
         [T + 2, 2,     T + 3, 3   ]],
    ).cuda()  # [b=2, s=4], mixed modalities on every rank

    result = head(hidden, labels=labels, current_step=0)

    assert result.total_loss is not None
    assert result.total_loss.isfinite(), f"Rank {global_rank}: total_loss not finite"
    assert result.lm_loss.isfinite().all(), f"Rank {global_rank}: lm_loss not finite"
    assert result.lm_loss.shape == (b, s)

    # ----- Routing + combine-A2A correctness check -----
    # flat_hidden and flat_labels share the same s-major flat index:
    #   flat[s_idx * b + b_idx] == hidden[s_idx, b_idx] / labels[b_idx, s_idx]
    flat_hidden = hidden.detach().reshape(s * b, h)
    flat_labels = labels.T.reshape(-1)          # [b,s] → [s,b] → [s*b]
    text_mask_flat = flat_labels < head_config.text_vocab_size
    va_mask_flat = ~text_mask_flat
    text_indices = text_mask_flat.nonzero(as_tuple=False).squeeze(-1)
    va_indices = va_mask_flat.nonzero(as_tuple=False).squeeze(-1)
    # [b,s] boolean masks for indexing lm_loss directly
    text_mask_bs = labels < head_config.text_vocab_size   # [b,s]
    va_mask_bs = ~text_mask_bs

    if ep_rank == 0:
        # Text expert: verify lm_loss at text positions matches this rank's hidden states.
        text_w = head.experts.projections["0"].weight.detach()
        tl = flat_hidden[text_indices] @ text_w.T
        ref_flat = torch.zeros(s * b, device="cuda")
        ref_flat[text_indices] = F.cross_entropy(tl, flat_labels[text_indices], reduction="none")
        ref_lm = ref_flat.view(s, b).T.contiguous()   # [b,s]
        assert torch.allclose(result.lm_loss[text_mask_bs], ref_lm[text_mask_bs], atol=1e-4, rtol=1e-4), (
            f"Rank {global_rank} (text expert): text positions in lm_loss don't match "
            f"hand-rolled reference from this rank's hidden states.\n"
            f"  result: {result.lm_loss[text_mask_bs]}\n"
            f"  ref:    {ref_lm[text_mask_bs]}"
        )
        # VA positions must be nonzero: combine A2A returned VA losses for this rank's VA tokens.
        assert result.lm_loss[va_mask_bs].abs().sum() > 0, (
            f"Rank {global_rank}: VA positions in lm_loss are all zero — "
            "combine A2A failed to return VA-expert losses for this rank's tokens"
        )
    else:
        # VA expert: verify lm_loss at VA positions matches this rank's hidden states.
        va_w = head.experts.projections["1"].weight.detach()
        vl = flat_hidden[va_indices] @ va_w.T
        ref_flat = torch.zeros(s * b, device="cuda")
        ref_flat[va_indices] = F.cross_entropy(
            vl, flat_labels[va_indices] - head_config.text_vocab_size, reduction="none"
        )
        ref_lm = ref_flat.view(s, b).T.contiguous()   # [b,s]
        assert torch.allclose(result.lm_loss[va_mask_bs], ref_lm[va_mask_bs], atol=1e-4, rtol=1e-4), (
            f"Rank {global_rank} (VA expert): VA positions in lm_loss don't match "
            f"hand-rolled reference from this rank's hidden states.\n"
            f"  result: {result.lm_loss[va_mask_bs]}\n"
            f"  ref:    {ref_lm[va_mask_bs]}"
        )
        # Text positions must be nonzero: combine A2A returned text losses for this rank's text tokens.
        assert result.lm_loss[text_mask_bs].abs().sum() > 0, (
            f"Rank {global_rank}: text positions in lm_loss are all zero — "
            "combine A2A failed to return text-expert losses for this rank's tokens"
        )

    # All 4 ranks must produce distinct total_loss values.
    loss_val = result.total_loss.detach().clone()
    all_losses = [torch.zeros_like(loss_val) for _ in range(world_size)]
    dist.all_gather(all_losses, loss_val)
    losses = torch.stack(all_losses)
    for i in range(world_size):
        for j in range(i + 1, world_size):
            assert not torch.allclose(losses[i], losses[j], atol=1e-6), (
                f"Ranks {i} and {j} produced identical total_loss despite different micro-batches"
            )

    # Backward must succeed with nonzero gradients on every rank.
    result.total_loss.backward()
    proj = head.experts.projections[str(ep_rank)]
    assert proj.weight.grad is not None, f"Rank {global_rank}: expert grad is None"
    assert proj.weight.grad.isfinite().all(), f"Rank {global_rank}: expert grad not finite"
    assert proj.weight.grad.abs().sum() > 0, f"Rank {global_rank}: expert grad is all-zero"
    assert hidden.grad is not None, f"Rank {global_rank}: hidden.grad is None"
    assert hidden.grad.isfinite().all(), f"Rank {global_rank}: hidden.grad not finite"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-vv", "-s"]))
