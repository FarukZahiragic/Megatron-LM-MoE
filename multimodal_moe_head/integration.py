"""Integration helpers.

This module is intentionally small. The goal is to construct a
`MoEOutputHead` and show how to plug it into a `GPTModel` *without modifying
any Megatron files*. The recommended pattern is to subclass `GPTModel` in your
own training code and override `forward` to call the head.

Example
-------

```python
from megatron.core.models.gpt.gpt_model import GPTModel
from multimodal_moe_head import (
    MoEOutputHead, MoEOutputHeadConfig, build_moe_output_head,
)


class MoEOutputGPTModel(GPTModel):
    def __init__(self, *args, head_config: MoEOutputHeadConfig, **kwargs):
        # Build the dense model. Important: set
        # `share_embeddings_and_output_weights=False` in the config so the
        # default tied output_layer is unused.
        super().__init__(*args, **kwargs)
        if self.post_process:
            # Replace the dense output_layer with our two-expert head.
            del self.output_layer
            self.output_layer = build_moe_output_head(
                self.config, head_config, tp_group=self.pg_collection.tp,
            )

    def forward(self, input_ids, position_ids, attention_mask,
                labels=None, **kwargs):
        # Run the base GPTModel up to the final hidden states.
        # The simplest way is to call the parent forward with `labels=None`
        # so it returns logits, but here we re-implement the post-process
        # tail to use our MoE head.
        ...  # left as an exercise; see the docstring for a full skeleton.
```

A complete reference subclass is intentionally not provided here because the
exact forward override depends on which Megatron release / fork you are on
(MTP, materialize_only_last_token_logits, etc.). The factory below is the
only piece of "wiring" this package owns.
"""

from typing import Optional

import torch

from megatron.core.transformer.transformer_config import TransformerConfig

from multimodal_moe_head.config import MoEOutputHeadConfig
from multimodal_moe_head.head import MoEOutputHead


def build_moe_output_head(
    config: TransformerConfig,
    head_config: MoEOutputHeadConfig,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> MoEOutputHead:
    """Construct an `MoEOutputHead` from the parent `TransformerConfig`.

    Args:
        config: the same TransformerConfig used to build the rest of the model.
            We read `hidden_size`, `params_dtype`, `init_method`, etc., from it.
        head_config: the head-specific configuration.
        tp_group: tensor-parallel process group for the expert projections.
            Pass `pg_collection.tp` from your model. If `None`, defaults to the
            globally-initialised TP group from `megatron.core.parallel_state`.

    Returns:
        A `MoEOutputHead` instance ready to consume hidden states of shape
        `[seq, batch, hidden_size]`.
    """
    return MoEOutputHead(config=config, head_config=head_config, tp_group=tp_group)
