"""Utilities for deep learning models."""

from enum import Enum, auto

import torch
import torch.nn.functional as tf
from torch import nn


class PredictionMode(Enum):
    """An enum for specifying a models prediction mode."""

    CLASSIFICATION = auto()
    REGRESSION = auto()


def add_last_layer(out: torch.Tensor, prediction_mode: PredictionMode) -> torch.Tensor:
    """Pass a torch tensor through the last layer to an internal deep learning model based on its prediction mode.

    Args:
        out: A torch tensor containing entries for the second-to-last layer of a neural network.
        prediction_mode: Prediction mode of the model

    Returns:
        torch.Tensor: The input layer passed through the final layer of the network

    """
    match prediction_mode:
        case PredictionMode.CLASSIFICATION:
            out = tf.softmax(out, dim=1)
        case PredictionMode.REGRESSION:
            pass
    return out


def compute_mse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute MSE of torch model."""
    return ((predictions.detach().cpu().numpy() - targets.detach().cpu().numpy()) ** 2).mean()


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute accuracy of torch model."""
    return (predictions.argmax(dim=1).numpy() == targets.numpy()).mean()


def freeze_layers(model: nn.Module, layers: list[str] | None = None) -> None:
    """Freeze all parameters in the model."""
    for name, param in model.named_parameters():
        if layers is not None and not any(layer_name in name for layer_name in layers):
            continue
        param.requires_grad = False


def freeze_until_layer(model: nn.Module, target_layer: str | None = None) -> None:
    """Freeze all layers until a specific layer."""
    reached_target = False
    for name, param in model.named_parameters():
        if target_layer and target_layer in name:
            reached_target = True
        if not reached_target:
            param.requires_grad = False
