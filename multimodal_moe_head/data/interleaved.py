"""Strict 1:1 modality interleaving for the MoE-head training runs.

Builds two INDEPENDENT blended GPT datasets — one over the text corpora, one
over the vision corpora — and interleaves them so that even global sample
indices come from text and odd indices come from vision:

    global idx:   0     1     2     3     4     5    ...
    served from:  text  vis   text  vis   text  vis  ...

Why this guarantees "1 text + 1 vision per micro-batch":

  Megatron's `single` dataloader uses MegatronPretrainingSampler, which consumes
  sample indices strictly in order (range(consumed, total)) and hands each
  data-parallel rank a CONTIGUOUS micro-batch-size slice of each global chunk
  (rank r -> indices [r*MBS, r*MBS+MBS)). With MBS=2 every rank therefore gets a
  pair of CONSECUTIVE indices [2k, 2k+1] -> exactly one text and one vision
  sample. This holds on every rank, every micro-batch, every step, and survives
  checkpoint/resume because index -> modality is a fixed function of the index.

Audio is intentionally ignored for now (only two modalities are interleaved).

Note: this only makes sense for MBS == 2. With MBS=2 each micro-batch is one
text + one vision sample; the module logs a warning for other MBS values since
the per-micro-batch balance would no longer be 1:1.
"""

import copy
from functools import partial
from typing import List, Optional, Tuple

import torch

from megatron.core.datasets.blended_megatron_dataset_builder import (
    BlendedMegatronDatasetBuilder,
)


class InterleavedModalityDataset(torch.utils.data.Dataset):
    """Interleave two datasets 1:1 (even idx -> text, odd idx -> vision)."""

    def __init__(self, text_dataset, vision_dataset):
        self.text_dataset = text_dataset
        self.vision_dataset = vision_dataset
        # Each output pair consumes one text and one vision sample, so the number
        # of pairs is bounded by the smaller side.
        self._pairs = min(len(text_dataset), len(vision_dataset))

    def __len__(self):
        return 2 * self._pairs

    def __getitem__(self, idx):
        pair = idx // 2
        if idx % 2 == 0:
            return self.text_dataset[pair]
        return self.vision_dataset[pair]


def build_interleaved_modality_datasets(
    dataset_type,
    train_val_test_num_samples,
    is_built_on_rank,
    base_config,
    text_blend: Tuple[List[str], Optional[List[float]]],
    vision_blend: Tuple[List[str], Optional[List[float]]],
    split: str,
):
    """Build a text blend and a vision blend, then interleave them 1:1.

    Returns (train, valid, test) where each is an InterleavedModalityDataset (or
    None when the corresponding split is empty, e.g. with --split 100,0,0).

    `split` is the real train/valid/test split string (e.g. "100,0,0"). It must
    be passed in because the base config was built without --data-path, which
    causes BlendedMegatronDatasetConfig.__post_init__ to set mock=True and
    silently override split to "1,1,1" (an even 3-way split) -- we restore the
    intended split below.
    """
    from megatron.core.datasets.blended_megatron_dataset_config import (
        convert_split_vector_to_split_matrix,
        parse_and_normalize_split,
    )

    split_matrix = convert_split_vector_to_split_matrix(parse_and_normalize_split(split))

    def _build_one(blend):
        cfg = copy.copy(base_config)
        cfg.blend = blend
        cfg.blend_per_split = None
        # The base config is constructed without --data-path (we supply the blends
        # here), so BlendedMegatronDatasetConfig.__post_init__ flips mock=True AND
        # overrides split -> "1,1,1" (even 3-way) + the matching split_matrix.
        # copy.copy does NOT re-run __post_init__, so restore all three explicitly;
        # otherwise the builder would (a) build a MockGPTDataset and (b) carve the
        # data into train/valid/test, with zero-width per-dataset splits crashing
        # build_sample_idx ("tokens_per_epoch > 1") on small/single-doc datasets.
        cfg.mock = False
        cfg.split = split
        cfg.split_matrix = split_matrix
        return BlendedMegatronDatasetBuilder(
            dataset_type, train_val_test_num_samples, is_built_on_rank, cfg
        ).build()

    text_train, text_valid, text_test = _build_one(text_blend)
    vis_train, vis_valid, vis_test = _build_one(vision_blend)

    def _interleave(text_ds, vis_ds):
        if text_ds is None or vis_ds is None:
            return None
        return InterleavedModalityDataset(text_ds, vis_ds)

    return (
        _interleave(text_train, vis_train),
        _interleave(text_valid, vis_valid),
        _interleave(text_test, vis_test),
    )
