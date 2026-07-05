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

> For the full story behind this module — the design decisions, training runs,
> dead-ends, and the eval/visualization tooling — see [`JOURNAL.md`](JOURNAL.md).

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
  the router's argmax. For this hard-routing head the recommended default is
  **always-on** (`1000000000`); turning it off only ever feeds misrouted tokens a
  garbage-target gradient and never helps the router (which trains from the aux
  loss either way). See the `moe-teacher-force-default` note for the full rationale.
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
data/                training data helpers
  mock.py              mock text+vision batches for smoke tests
  interleaved.py       1-text + 1-vision micro-batch interleaving for real data
tests/               EP=1 (replicated) and EP=2 correctness tests
scripts/             SLURM training launchers (submit-*.sh, see below)
retokenize/          re-tokenize the ClimbMix corpus with the Apertus tokenizer
hf_export/           convert an EP=2 MoE checkpoint to an HF ApertusMoE model (see CONVERSION.md)
```

## How it plugs into Megatron

The head is wired into the stock GPT pipeline **without patching core files** in
ways that affect dense training — the hooks live in the repo-root trainer:

- `pretrain_gpt.py` — `add_moe_head_args()` (the `--moe-output-head`,
  `--moe-teacher-force-steps`, `--moe-router-loss-coeff` flags) and
  `_adaptive_gpt_builder()` (selects `moe_gpt_builder` when `--moe-output-head`).
- `gpt_builders.py` — `moe_gpt_builder()` constructs an `MoEGPTModel`.

`MoEGPTModel` returns a `[b, s]` per-token loss — exactly the shape
`loss_func` in `pretrain_gpt.py` already expects — so the rest of the training
loop is unchanged.

## Training

Each launcher is a self-contained `sbatch`-able script. The `1b-climbmix` pair is
the current Stage-1 setup (32 nodes, GBS 256, WSD LR):

```bash
# dense baseline
sbatch multimodal_moe_head/scripts/submit-apertus-1p5-1b-climbmix-dense-stage1.sh
# MoE head (EP=2)
sbatch multimodal_moe_head/scripts/submit-apertus-1p5-1b-climbmix-moe-stage1.sh
```

The dense/moe scripts differ only by the `USE_MOE_HEAD` default; both honour an
override (`USE_MOE_HEAD=true|false`).

| Script | Purpose |
|--------|---------|
| `submit-apertus-1p5-1b-climbmix-dense-stage1.sh` | 1B dense baseline, ClimbMix |
| `submit-apertus-1p5-1b-climbmix-moe-stage1.sh`   | 1B MoE head (EP=2), ClimbMix |
| `submit-apertus-300m-climbmix-stage1.sh`         | 300M ClimbMix |
| `submit-apertus-1p5-1b-dense-text.sh` / `...-300m-dense-text.sh` | earlier text-only runs |
| `submit-apertus-1p5-1b-moe-head-stage1.sh`       | earlier 1B MoE run |
| `submit-apertus-300m-moe-head-stage1.sh`         | 300M MoE Stage-1 |
| `submit-apertus-300m{,-moe-head,-standard-head}-mock.sh`, `submit-apertus-8b-*-mock.sh` | mock-data smoke tests (no real dataset) |
| `submit-apertus-300m.sh` | **production dense reference — do not edit** |

## Tests

```bash
bash multimodal_moe_head/tests/run_test_replicated.sh   # EP=1
bash multimodal_moe_head/tests/run_test_ep2.sh          # EP=2 (2 ranks)
```

## Data tooling

- `retokenize/` — re-tokenize ClimbMix (GPT-2 parquet → Apertus tokens + EOS).
  `_run_full.sbatch` runs all 100 shards (≈32.7B Apertus tokens).
- `hf_export/` — convert an EP=2 MoE dist-checkpoint to an HF `ApertusMoEForCausalLM`
  so it can be scored by lm-evaluation-harness. Full procedure in
  [`hf_export/CONVERSION.md`](hf_export/CONVERSION.md); `_convert_1b_moe_62k.sbatch`
  (MoE) and `_gen_1b_dense_62k.sbatch` (dense) are the current job scripts.
