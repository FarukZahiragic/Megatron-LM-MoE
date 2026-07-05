import torch
import torch.nn.functional as F


def modality_target_from_labels(labels: torch.Tensor, text_vocab_size: int) -> torch.Tensor:
    """Derive the ground-truth modality (0 = text, 1 = vision/audio) from labels.

    Args:
        labels: integer tensor of any shape with token ids.
        text_vocab_size: size of the text vocab (boundary between modalities).

    Returns:
        torch.LongTensor of the same shape with values in {0, 1}.
    """
    return (labels >= text_vocab_size).long()


def router_modality_loss(
    router_logits: torch.Tensor,
    modality_target: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross-entropy between router predictions and the ground-truth modality.

    Args:
        router_logits: [N, 2] router output (any dtype; cast to float32 internally).
        modality_target: [N] long tensor in {0, 1, ignore_index}.
        ignore_index: positions to skip (typically -100 for padding).

    Returns:
        Scalar mean loss over non-ignored positions.
    """
    return F.cross_entropy(
        router_logits.float(),
        modality_target,
        ignore_index=ignore_index,
        reduction="mean",
    )


def teacher_forced_routing_map(modality_target: torch.Tensor) -> torch.Tensor:
    """Build a one-hot routing map [N, 2] from ground-truth modality.

    Positions with ignore_index (negative values) are routed to expert 0 by
    convention; the caller should mask out the corresponding loss anyway.
    """
    n = modality_target.shape[0]
    clamped = modality_target.clamp(min=0, max=1)
    routing_map = torch.zeros((n, 2), dtype=torch.bool, device=modality_target.device)
    routing_map[torch.arange(n, device=modality_target.device), clamped] = True
    return routing_map


