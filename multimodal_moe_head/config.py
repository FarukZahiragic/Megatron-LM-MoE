from dataclasses import dataclass


@dataclass
class MoEOutputHeadConfig:
    """Configuration for the two-expert modality-routed output head.

    The output vocabulary is partitioned into two contiguous ranges:
        text:        [0,                text_vocab_size)
        vision/audio:[text_vocab_size,  total_vocab_size)

    A learned top-1 router predicts which range the next token belongs to.
    """

    total_vocab_size: int
    text_vocab_size: int = 131072

    router_loss_coeff: float = 0.01

    teacher_force_steps: int = 0

    expert_parallel_size: int = 1

    # Diagnostic: log the per-sequence modality of the first N micro-batches this
    # rank receives (0 = off). Used to verify the 1-text + 1-vision interleaving.
    log_router_inputs: int = 0

    def __post_init__(self):
        assert self.text_vocab_size > 0
        assert self.total_vocab_size > self.text_vocab_size, (
            f"total_vocab_size ({self.total_vocab_size}) must exceed "
            f"text_vocab_size ({self.text_vocab_size})"
        )
        assert self.expert_parallel_size in (1, 2), (
            "Only EP=1 (replicated) and EP=2 (one expert per rank) are supported."
        )

    @property
    def va_vocab_size(self) -> int:
        return self.total_vocab_size - self.text_vocab_size

    @property
    def expert_vocab_sizes(self) -> list[int]:
        return [self.text_vocab_size, self.va_vocab_size]
