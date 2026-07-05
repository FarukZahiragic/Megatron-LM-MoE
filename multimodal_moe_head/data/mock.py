"""Synthetic multimodal mock data for exercising the two-expert MoE output head.

Megatron's stock ``MockGPTLowLevelDataset`` emits a sequential ramp
``[1, 2, 3, ..., length-1, eod]``. Every token id is tiny, so all of them fall
below ``base_vocab_size`` (the text/VA boundary) and route to the text expert —
the vision/audio expert never sees a token. That is fine for checking the
pipeline runs, but it never exercises the VA projection, the router under a real
mix, or a balanced expert utilization.

``enable_multimodal_mock`` monkeypatches the low-level mock dataset so each
generated document is a random mix of text tokens (ids in ``[1, base_vocab)``)
and vision/audio tokens (ids in ``[base_vocab, padded_vocab)``), roughly
``text_fraction`` text. The vocab boundaries are read from the global args at
call time, so no pre-tokenized data or knowledge of the exact vocab sizes is
needed. Token assignment is deterministic per sample index (seeded by the
index) so runs are reproducible.

Activated by ``--moe-mock-multimodal`` (see pretrain_gpt.add_moe_head_args),
applied inside the dataset provider before datasets are built.
"""

import numpy

from megatron.core.datasets.gpt_dataset import MockGPTLowLevelDataset
from megatron.training import get_args, print_rank_0

_PATCHED = False


def enable_multimodal_mock(text_fraction: float = 0.5) -> None:
    """Patch MockGPTLowLevelDataset to emit a text/VA token mix.

    Args:
        text_fraction: expected fraction of tokens drawn from the text range.
            The rest are drawn from the vision/audio range.
    """
    global _PATCHED
    if _PATCHED:
        return

    def _multimodal_getitem(self, idx):
        args = get_args()
        # Prefer an omni tokenizer's base_vocab_size; otherwise fall back to the
        # explicit --moe-text-vocab-size boundary (same resolution as moe_gpt_builder).
        base_vocab = getattr(args, "base_vocab_size", None) or getattr(
            args, "moe_text_vocab_size", None
        )
        total_vocab = getattr(args, "padded_vocab_size", None)
        assert base_vocab is not None, (
            "--moe-mock-multimodal needs a text/VA boundary: either "
            "args.base_vocab_size (from an omni tokenizer) or --moe-text-vocab-size."
        )
        assert total_vocab is not None and total_vocab > base_vocab, (
            f"Need padded_vocab_size ({total_vocab}) > base_vocab_size "
            f"({base_vocab}) for a non-empty vision/audio range."
        )

        n_tokens = int(self.sequence_lengths[idx]) - 1
        if n_tokens <= 0:
            return numpy.array([self.tokenizer.eod], dtype=numpy.int64)

        # Deterministic per-index so the run is reproducible across ranks/restarts.
        rng = numpy.random.default_rng(seed=int(idx) + 1)
        is_text = rng.random(n_tokens) < text_fraction
        text_tokens = rng.integers(low=1, high=base_vocab, size=n_tokens)
        va_tokens = rng.integers(low=base_vocab, high=total_vocab, size=n_tokens)
        tokens = numpy.where(is_text, text_tokens, va_tokens)

        sample = numpy.concatenate([tokens, [self.tokenizer.eod]]).astype(numpy.int64)
        return sample

    MockGPTLowLevelDataset.__getitem__ = _multimodal_getitem
    _PATCHED = True
    print_rank_0(
        f"> [moe-head] multimodal mock data enabled "
        f"(~{text_fraction:.0%} text / {1 - text_fraction:.0%} vision-audio per sequence)"
    )
