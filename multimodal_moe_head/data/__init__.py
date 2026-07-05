"""Data helpers for MoE-head training.

- `mock`: synthetic text/vision-audio mock batches for smoke tests.
- `interleaved`: strict 1-text + 1-vision micro-batch interleaving for real data.
"""

from multimodal_moe_head.data.interleaved import (
    InterleavedModalityDataset,
    build_interleaved_modality_datasets,
)
from multimodal_moe_head.data.mock import enable_multimodal_mock

__all__ = [
    "InterleavedModalityDataset",
    "build_interleaved_modality_datasets",
    "enable_multimodal_mock",
]
