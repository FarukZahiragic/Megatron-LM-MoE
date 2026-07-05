from typing import List, Optional

import torch
from torch import nn

from megatron.core import tensor_parallel
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig


class OutputProjectionExperts(MegatronModule):
    """Two heterogeneous output projections.

    expert 0 -> [hidden_size, text_vocab_size]
    expert 1 -> [hidden_size, va_vocab_size]

    Each projection is a ColumnParallelLinear sharded across the given TP group.
    The output of each projection is left sharded (gather_output=False) so that
    vocab_parallel_cross_entropy can consume it directly.

    When local_expert_indices contains both 0 and 1 (replicated / EP=1), both
    projections are allocated on this rank. When EP=2, this rank only allocates
    the projection for its assigned expert.
    """

    def __init__(
        self,
        config: TransformerConfig,
        vocab_sizes: List[int],
        local_expert_indices: List[int],
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        expert_parallel: bool = False,
    ):
        super().__init__(config=config)
        assert len(vocab_sizes) == 2, "Exactly two experts (text, vision/audio) are expected."
        self.vocab_sizes = vocab_sizes
        self.expert_parallel = expert_parallel
        self.local_expert_indices = list(local_expert_indices)
        for idx in self.local_expert_indices:
            assert idx in (0, 1), f"expert index {idx} out of range"

        self.projections = nn.ModuleDict()
        for idx in self.local_expert_indices:
            self.projections[str(idx)] = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                vocab_sizes[idx],
                config=config,
                init_method=config.init_method,
                bias=False,
                gather_output=False,
                skip_bias_add=False,
                tp_group=tp_group,
            )

    def has_expert(self, expert_idx: int) -> bool:
        return str(expert_idx) in self.projections

    def forward(self, expert_idx: int, hidden: torch.Tensor) -> torch.Tensor:
        """Run the given expert's projection on `hidden`.

        Args:
            expert_idx: 0 (text) or 1 (vision/audio).
            hidden: [N, H] tensor of hidden states routed to this expert.

        Returns:
            Logits of shape [N, vocab_sizes[expert_idx] / tp_size].
        """
        assert self.has_expert(expert_idx), (
            f"expert {expert_idx} is not allocated on this rank "
            f"(local_expert_indices={self.local_expert_indices})"
        )
        proj = self.projections[str(expert_idx)]
        logits, _ = proj(hidden)
        return logits

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Distributed-checkpoint sharding for the (possibly expert-parallel) projections.

        Each projection is a full (TP-sharded) tensor under a unique key
        (`projections.0.weight`, `projections.1.weight`). The default
        ColumnParallelLinear implementation stamps the tensor's `replica_id`
        DP coordinate from the *dense* data-parallel rank. Under EP>1 that is
        wrong: the VA expert lives only on EP rank 1 (dense DP rank 1), so it
        gets replica_id=(0, 0, 1) — i.e. it looks like data-parallel replica #1
        of a tensor whose primary (replica 0) exists on no rank, and
        dist-checkpoint validation rejects it ("Invalid access pattern").

        Each expert is actually the sole owner of its weight within its
        expert-data-parallel group, so we re-stamp the DP coordinate with the
        expert-data-parallel rank (0 for the owner). Under EP=1 the experts are
        replicated across the normal DP group and the default replica_id is
        already correct, so we leave it untouched.
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)

        if self.expert_parallel:
            from megatron.core import parallel_state

            edp_rank = parallel_state.get_expert_data_parallel_rank()
            for sh_ten in sharded_state_dict.values():
                replica_id = getattr(sh_ten, 'replica_id', None)
                if replica_id is not None and len(replica_id) == 3:
                    sh_ten.replica_id = (*replica_id[:2], edp_rank)

        return sharded_state_dict
