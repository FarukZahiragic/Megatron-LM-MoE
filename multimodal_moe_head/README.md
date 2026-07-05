# multimodal_moe_head

A two-expert, **modality-routed output head** for Apertus / Megatron-LM. It
replaces the standard full-vocab `ColumnParallelLinear` LM head with a learned
top-1 router that sends each token to one of two output projections:

| Expert | Vocab range | Tokens |
|--------|-------------|--------|
| 0 — text          | `[0, text_vocab_size)`              | text |
| 1 — vision/audio  | `[text_vocab_size, total_vocab_size)` | image / audio |

The router is a `Linear(hidden, 2)` trained by an **auxiliary cross-entropy**
against the ground-truth modality (derived from the label token-id range). The
LM loss is the usual per-token cross-entropy, computed *within* the chosen
expert's vocab via `vocab_parallel_cross_entropy`.

## Background

This package is a port of **Method II** from a semester project: a two-expert
modality-routed output head that materializes logits for only one modality per
token (~50% output-layer compute/memory under EP=2). In a prior 1B climb-mix
ablation (dense backbone + this head), per-rank peak reserved memory dropped
~34% (~16 GB) and mean iteration time ~9%, with LM/text/vision losses tracking
the dense baseline.

It is intended as an **opt-in baseline for Apertus 2** — gated behind
`--moe-output-head` so dense training is unchanged when the flag is off.

## Design at a glance

- **Hard top-1 routing (argmax).** The dispatch is non-differentiable, so the
  router receives **no** LM gradient — it is trained solely by the auxiliary
  `router_modality_loss`, which is computed for every token regardless of how it
  was dispatched. That aux loss also backprops into the backbone, pushing hidden
  states to be modality-separable.
- **Auxiliary loss attachment** (`moe_gpt_model.py:AuxLossAutoScaler`). A custom
  autograd `Function` that returns the LM loss unchanged in forward and fires
  `router_loss.backward(1.0)` in backward — so the router trains without
  inflating the reported LM-loss scalar. Mirrors Megatron's `MTPLossAutoScaler`.
- **Teacher forcing** (`teacher_force_steps`). While `current_step <
  teacher_force_steps`, tokens are dispatched by ground-truth modality instead of
  the router's argmax. Default is **always-on** (`1000000000`).
- **Expert parallelism.**
  - `EP=1` (replicated): both projections on every rank; each rank handles all
    its tokens locally. Default, safest.
  - `EP=2` (one expert per rank): text expert on EP rank 0, VA expert on EP rank
    1; tokens exchanged with a pair of all-to-alls. `experts.py:sharded_state_dict`
    re-stamps the dist-checkpoint `replica_id` so EP=2 checkpoints save/load
    correctly.

## Module layout

```
config.py            MoEOutputHeadConfig (vocab split, router coeff, EP size, ...)
router_loss.py       modality target + router CE + teacher-forced routing map
experts.py           OutputProjectionExperts: the two ColumnParallelLinear heads
head.py              MoEOutputHead: routing, all-to-all (EP=2), per-token loss
moe_gpt_model.py     MoEGPTModel (GPTModel subclass) + AuxLossAutoScaler
integration.py       build_moe_output_head() factory + integration notes
data/
  mock.py              mock text+vision batches for smoke tests
  interleaved.py       1-text + 1-vision micro-batch interleaving for real data
tests/               EP=1 and EP=2 GPU correctness tests (see below)
```

## How it plugs into Megatron

The head is wired into the stock GPT pipeline **without patching core
`megatron/` files**:

- `pretrain_gpt.py` — `add_moe_head_args()`, `_adaptive_gpt_builder()`, router
  metrics in `loss_func`, optional `--moe-mock-multimodal` / interleaved data hooks.
- `gpt_builders.py` — `moe_gpt_builder()` constructs an `MoEGPTModel`. Keeps the
  backbone dense by nulling `config.num_moe_experts` only around the layer-spec
  build; `--num-experts 2` still creates EP groups for the output head.

`MoEGPTModel` returns a `[b, s]` per-token loss — the shape `loss_func` already
expects — so the rest of the training loop is unchanged.

## Usage

```bash
# EP=2 (one expert per rank)
--moe-output-head --num-experts 2 --expert-model-parallel-size 2 \
  --moe-text-vocab-size 131072 --moe-router-loss-coeff 0.01 \
  --moe-teacher-force-steps 1000000000

# EP=1 (both experts replicated per rank) — omit --expert-model-parallel-size

# Mock multimodal smoke data (with --mock-data)
--moe-mock-multimodal
```

Modality boundary: `--moe-text-vocab-size` (default 131072), or `base_vocab_size`
from an omni tokenizer if present.

## Tests

GPU correctness tests live in `tests/`. Run from the **repo root** on a GPU node
(CUDA + NCCL required):

```bash
cd /path/to/Megatron-LM-MoE
export PYTHONPATH=$PWD

# EP=1 smoke (1 GPU)
RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29501 \
  python -m pytest multimodal_moe_head/tests/test_replicated.py -v -s

# EP=2 full suite (4 GPUs, EP=2 + DP=2)
torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29502 \
  -m pytest multimodal_moe_head/tests/test_ep2.py -v -s
```

### Verified (Clariden, GH200)

| Suite | GPUs | Result |
|-------|------|--------|
| `test_replicated.py` | 1 | 7/7 passed |
| `test_ep2.py` | 4 | 19/19 passed |

These tests verify routing, numerical loss parity (hand-rolled CE reference),
EP=2 all-to-all dispatch/combine, and gradient flow. They exercise
`MoEOutputHead` directly — not the full `pretrain_gpt.py` training loop.

### Optional follow-up

A short end-to-end `pretrain_gpt.py` run with `--moe-output-head --mock-data
--moe-mock-multimodal` would additionally confirm model build, `_postprocess`
integration, and metric logging in the real trainer. Not required for the baseline.

## Known limitations

- **Generation** not implemented (`labels=None` raises); training/eval-with-labels only.
- **MTP** not supported (assert in `MoEGPTModel`).
- **Load balance:** EP=2 with one text + one VA expert idles the VA rank under
  ~90/10 text/multimodal mixtures. Generalizing to more text experts or EP=1
  for skewed mixes is a natural follow-up.
- **Data pipeline:** full omni tokenizer / interleaved multimodal loaders are not
  ported in this fork; mock + interleaved hooks are included for testing.
