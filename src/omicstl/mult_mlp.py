"""A non-variational version of Deep-IMV."""
from collections.abc import Callable
import torch
import torch.nn.functional as tf
from torch import nn
from typing import Callable, List
from omicstl.deep_learning_utils import PredictionMode, add_last_layer, freeze_layers



class SimpleFC(nn.Module):
    """A simple MLP with batch normalization for a single view of the multi-view model.

    Attributes:
            input_size: The number of input features.
            hidden_sizes: A list with the number of hidden units in each hidden layer.
            prediction_dim: The number of output classes.
            prediction_mode: The mode of prediction for the output layer.
            activation_fn: The activation function used after each layer.
            dropout: The dropout rate applied after each activation.

    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        prediction_dim: int,
        prediction_mode: PredictionMode,
        activation_fn: Callable = tf.relu,
        dropout: float = 0.2,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the MLP with batch normalization and dropout.

        Args:
                input_size: Number of input features.
                hidden_sizes: List of hidden layer sizes.
                prediction_dim: Number of output classes.
                prediction_mode: Mode of the output layer (e.g., classification).
                activation_fn: Activation function to use (default: ReLU).
                dropout: Dropout rate (default: 0.2).
                device: Optional device to move the model to.

        """
        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.prediction_dim = prediction_dim
        self.prediction_mode = prediction_mode
        self.activation_fn = activation_fn

        self.fc_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        # First layer
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.fc_layers.append(nn.Linear(prev_size, hidden_size))
            self.bn_layers.append(nn.BatchNorm1d(hidden_size))
            prev_size = hidden_size

        self.fc_out = nn.Linear(hidden_sizes[-1], prediction_dim)
        self.dropout = nn.Dropout(dropout)

        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward method for SimpleFC."""
        h = x
        for fc, bn in zip(self.fc_layers, self.bn_layers, strict = True):
            h = self.activation_fn(bn(fc(h)))
            h = self.dropout(h)

        out = self.dropout(self.fc_out(h))
        out = add_last_layer(out, self.prediction_mode)

        return out, h


class SimpleFCHook(SimpleFC):
    """A simple MLP with batch norm and hooks to retrieve intermediate activations and gradients."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        prediction_dim: int,
        prediction_mode: PredictionMode,
        activation_fn: Callable = tf.relu,
        dropout: float = 0.2,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the MLP with hooks for capturing activations and gradients."""
        super().__init__(
            input_size,
            hidden_sizes,
            prediction_dim,
            prediction_mode,
            activation_fn,
            dropout,
            device,
        )

        self.activations: dict = {}
        self.activations_grad: dict = {}

        self.fc_layers[0].register_forward_hook(self.get_activation("fc1"))

    def get_activation(self, name: str) -> Callable:
        """Get actication function for FCHook model."""
        def hook(model, input, output):
            self.activations[name] = output

        return hook

    def get_activation_grad(self, name: str) -> Callable:
        """Get actication gradient for FCHook model."""
        def hook(grad):
            self.activations_grad[name] = grad

        return hook

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for simple FC with hooks."""
        h = x

        for i, (fc, bn) in enumerate(zip(self.fc_layers, self.bn_layers, strict = True)):
            h = self.activation_fn(bn(fc(h)))
            h = self.dropout(h)
            if i == 0:
                h.register_hook(self.get_activation_grad("fc1"))

        out = self.dropout(self.fc_out(h))
        out = add_last_layer(out, self.prediction_mode)

        return out, h


class JointMLP(nn.Module):
    """A model that fuses the predictions of multiple marginal models.

    Attributes:
            margin_models: A list of marginal models of type simple_FC
            fc1: The first fully connected layer after the last hidden layer of the marginal models
            fc2: The second fully connected layer, immediately after fc1
            dropout: A dropout layer
            device: The device to run the model on

    """

    def __init__(
        self,
        marginal_models: list[SimpleFC],
        hidden_dim: int = 128,
        activation_fn: Callable = tf.relu,
        dropout: float = 0.2,
        combine_fn: str = "mean",
        hooks: bool = False,
        device: torch.device | None = None,
    ) -> None:
        """Initialize a model that fuses the predictions of multiple marginal models.

        Args:
                marginal_models: A list of marginal models of type simple_FC
                hidden_dim: The number of hidden units between fc1 and fc2. Defaults to 128.
                activation_fn: The activation function of the hidden layers.
                dropout: The dropout rate. Defaults to 0.2.
                combine_fn: How the views are combined in the latent space.
                hooks: TODO: what does this do?
                device: The device to run the model on. Defaults to None.

        """
        super().__init__()

        self.device = device
        prediction_modes = [model.prediction_mode for model in marginal_models]
        if all(mode == prediction_modes[0] for mode in prediction_modes):
            self.prediction_mode = prediction_modes[0]
        else:
            raise RuntimeError("Marginal models do not have the same prediction mode.")

        self.margin_models = torch.nn.ModuleList(marginal_models)

        if combine_fn == "mean":
            dim_fc1_in = marginal_models[0].hidden_sizes[-1]
        elif combine_fn == "concat":
            dim_fc1_in = sum([m.hidden_sizes[-1] for m in self.margin_models])

        self.combine_fn = combine_fn
        self.fc1 = nn.Linear(dim_fc1_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.margin_models[0].prediction_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation_fn = activation_fn
        self.hooks = hooks

        if self.hooks:
            self.activations: dict = {}
            self.activations_grad: dict = {}

            self.fc_layers[0].register_forward_hook(self.get_activation("fc1"))

        if device is not None:
            self.to(device)

    def get_activation(self, name):
        def hook(model, input, output):
            self.activations[name] = output

        return hook

    def get_activation_grad(self, name):
        def hook(grad):
            self.activations_grad[name] = grad

        return hook

    def forward(
        self,
        x: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        if len(x) == 1 and isinstance(x[0], list):
            x = x[0]  # Unpack the single list of tensors
        assert len(x) == len(self.margin_models), "Number of inputs must match number of marginal models"

        yhats = []
        hiddens = []

        # maintain a queue of poe_dists
        # separate the input tensors into batches with 1, 2, 3, ... complete views
        # for each batch, compute the distributions and fine the poe_dist for that batch, append it to the queue of poe_dists
        # once you've gone through all batches
        # Nono, do the batching thing outside this loop, but accumulate the losses (add em up), then call backwards() on the sum

        for i, model in enumerate(self.margin_models):
            # ignore view-missing data
            if not isinstance(x[i], torch.Tensor):
                print("x not tensor")
                # TODO: Add raise here
                continue

            # Move input to device if needed
            if self.device is not None and x[i].device != self.device:
                x[i] = x[i].to(self.device)

            yhat, h = model(x[i])
            yhats.append(yhat)
            hiddens.append(h)

        if self.combine_fn == "mean":
            h = torch.mean(torch.stack(hiddens), dim=0)
        elif self.combine_fn == "concat":
            h = torch.cat(hiddens, dim=-1)

        h = self.dropout(self.activation_fn(self.fc1(h)))

        if self.hooks:
            h.register_hook(self.get_activation_grad("fc1"))

        yhat = add_last_layer(self.fc2(h), self.prediction_mode)

        return yhat, h, yhats, hiddens

    def loss(
        self,
        y: torch.Tensor,
        yhat: torch.Tensor,
        yhats: list[torch.Tensor],
        focal: bool = False,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        marginal_weight: float | None = None,
        marginal_coefs: list[float] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """Compute the total loss for all the joint and marginal models.

        Args:
                y: Ground truth labels
                yhat: softmax predictions for the combinations of experts
                yhats: List of softmax predictions for each marginal model
                focal: Whether to use focal loss. Defaults to True.
                gamma: The focal loss gamma parameter. Defaults to 2.
                alpha: A tensor with number of elements equal to the number of classes, specifying class weights. Defaults to None.
                marginal_weight: TODO: What does this do?
                marginal_coefs: TODO: What does this do?

        Returns:
                tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]: The joint loss

        """
        # Move data to device if needed
        if self.device is not None:
            if y.device != self.device:
                y = y.to(self.device)
            if yhat.device != self.device:
                yhat = yhat.to(self.device)

        match self.prediction_mode:
            case PredictionMode.CLASSIFICATION:
                product_loss = tf.cross_entropy(yhat, y, reduction="none")
                marginal_losses = [tf.cross_entropy(yh, y, reduction="none") for yh in yhats]

                if alpha is not None:
                    if self.device is not None and alpha.device != self.device:
                        alpha = alpha.to(self.device)
                    alpha = alpha.repeat(yhat.shape[0], 1)
                    alpha = alpha.gather(1, y.view(-1, 1))
                    product_loss = product_loss * alpha.view(-1)
                    marginal_losses = [m * alpha.view(-1) for m in marginal_losses]

                if focal:
                    product_loss = torch.pow(1 - yhat.gather(1, y.view(-1, 1)), gamma).view(-1) * product_loss
                    marginal_losses = [
                        torch.pow(1 - yh.gather(1, y.view(-1, 1)), gamma).view(-1) * m
                        for yh, m in zip(yhats, marginal_losses, strict=False)
                    ]

            case PredictionMode.REGRESSION:
                product_loss = tf.mse_loss(yhat.view(-1), y, reduction="none")
                marginal_losses = [tf.mse_loss(yh.view(-1), y, reduction="none") for yh in yhats]

        product_loss = torch.mean(product_loss)
        marginal_losses = [torch.mean(m) for m in marginal_losses]

        if marginal_weight is not None:
            if marginal_coefs is None:
                marginal_coefs = [1.0] * len(marginal_losses)
            marginal_losses = [coef * loss for coef, loss in zip(marginal_coefs, marginal_losses, strict=False)]

            avg_marginal_loss = sum(marginal_losses) / len(marginal_losses)

            loss = product_loss + marginal_weight * avg_marginal_loss
        else:
            loss = product_loss + sum(marginal_losses) / len(marginal_losses)

        return product_loss, marginal_losses, loss

    def _match_dims(self, y_a: torch.Tensor, y_b: torch.Tensor) -> torch.Tensor:
        y_expanded = y_b.view(*y_b.shape, *([1] * (y_a.dim() - y_b.dim())))
        return y_expanded.expand_as(y_a)

    def freeze_marginal_layers(self) -> None:
        """Freeze marginal layers."""
        for model in self.margin_models:
            freeze_layers(model)

    def freeze_joint_layers(self) -> None:
        """Freeze joint layers."""
        freeze_layers(self, ["fc1"])


def make_joint_model(
    data_dim: list[int],
    prediction_dim: int,
    hidden_sizes: list[list[int]],
    dropout: float,
    hidden_dim: int,
    prediction_mode: PredictionMode,
    activation_fn: Callable = tf.relu,
    combine_fn: str = "concat",
    device: torch.device | None = None,
) -> JointMLP:
    """Create a joint model for multiple views. Each view gets its own 'marginal model', and then there is a fusion model that takes the output of each marginal model and combines them.

    Args:
            data_dim: A list of input dimensions, each representing a view
            prediction_dim: The number of output classes
            hidden_sizes: A list of hidden sizes for each marginal model
            dropout: The dropout rate
            hidden_dim: The number of hidden units between fc1 and fc2 of the combination model.
            activation_fn: The activation function
            combine_fn: The method to combine the marginal models. Defaults to 'concat'.
            device: The device to run the model on. Defaults to None.

    Returns:
            JointMLP: The joint model

    """
    marginal_models = []

    for k in range(len(data_dim)):
        input_size = data_dim[k]
        mmod = SimpleFC(
            input_size=input_size,
            hidden_sizes=hidden_sizes[k],
            prediction_dim=prediction_dim,
            prediction_mode=prediction_mode,
            dropout=dropout,
            activation_fn=activation_fn,
            device=device,
        )
        marginal_models.append(mmod)

    # joint model
    return JointMLP(
        marginal_models=marginal_models,
        hidden_dim=hidden_dim,
        activation_fn=activation_fn,
        combine_fn=combine_fn,
        device=device,
    )
