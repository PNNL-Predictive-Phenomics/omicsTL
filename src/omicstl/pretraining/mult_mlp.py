"""A non-variation version of Deep IMV which can be pretrained."""

import logging
import torch
from torch import nn

logger = logging.getLogger(__name__)

class ViewEncoder(nn.Module):
    """Encoder for a single view."""

    def __init__(
        self,
    ) -> None:
        """Create encoder for a single view."""
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass."""
        return x

    def loss(
        self,
    ) -> torch.Tensor:
        """Compute loss."""


class ViewDecoder(nn.Module):
    """Decoder for a single view."""

    def __init__(
        self,
    ) -> None:
        """Create decoder for a single view."""
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass."""
        return x

    def loss(
        self,
    ) -> torch.Tensor:
        """Compute loss."""

class JointEncoder(nn.Module):
    """Decoder for the joint model."""

    def __init__(
        self,
    ) -> None:
        """Create encoder for the joint model."""
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass."""
        return x

    def loss(
        self,
    ) -> torch.Tensor:
        """Compute loss."""

class JointDecoder(nn.Module):
    """Decoder for the joint model."""

    def __init__(
        self,
    ) -> None:
        """Create decoder for the joint model."""
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass."""
        return x

    def loss(
        self,
    ) -> torch.Tensor:
        """Compute loss."""
