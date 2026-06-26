import torch
import torch.nn.functional as tf
from torch import nn
from torch.distributions import Normal, kl_divergence

from omicstl.deep_learning_utils import PredictionMode, add_last_layer, freeze_layers


class MarginalFC(nn.Module):
    """A simple MLP for a single view of the multi-view model.

    Attributes:
        input_size: The number of input features.
        hidden_sizes: The number of hidden units in each layer.
        z_dim: The dimensionality of the latent representation.
        prediction_dim: The number of output classes.
        dropout: The dropout layer applied after each hidden layer.
        fc1: The first fully connected layer after the input.
        fc_z: The layer that maps the last hidden layer to the latent distribution parameters.
        fc_out: The layer that maps the sampled latent representation to the output classes.
        device: The device to run the model on.

    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        z_dim: int,
        prediction_dim: int,
        prediction_mode: PredictionMode,
        dropout: float = 0.2,
        device: torch.device | None = None,
    ) -> None:
        """Initialize a simple MLP for a single view of the multi-view model.

        Args:
            input_size: The number of input features.
            hidden_sizes: The number of hidden units in each layer.
            z_dim: The dimensionality of the latent representation.
            prediction_dim: The number of output classes.
            dropout: The dropout rate. Defaults to 0.2.
            device: The device to run the model on. Defaults to None.

        """
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.z_dim = z_dim
        self.prediction_dim = prediction_dim
        self.prediction_mode = prediction_mode
        self.device = device

        self.fc1 = nn.Linear(input_size, hidden_sizes[0])

        for sz in range(1, len(hidden_sizes)):
            setattr(self, f"fc{sz + 1}", nn.Linear(hidden_sizes[sz - 1], hidden_sizes[sz]))

        self.fc_z = nn.Linear(hidden_sizes[-1], z_dim * 2)
        self.fc_out = nn.Linear(z_dim, prediction_dim)
        self.dropout = nn.Dropout(dropout)

        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Normal]:
        """Forward method for marginal fc."""
        if self.device is not None and x.device != self.device:
            x = x.to(self.device)

        h = tf.relu(self.fc1(x))

        for sz in range(1, len(self.hidden_sizes)):
            h = self.dropout(tf.relu(getattr(self, f"fc{sz + 1}")(h)))

        h = self.fc_z(h)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        logvar = torch.clamp(logvar, -10, 10)

        dist = Normal(mu, torch.exp(logvar / 2))

        z = dist.rsample()
        out = self.dropout(self.fc_out(z))
        out = add_last_layer(out, self.prediction_mode)

        return out, dist


class JointVAE(nn.Module):
    """A joint VAE model for integrating multiple views.

    Attributes:
        margin_models: The list of marginal models of class FC_Marginal
        hidden_dim: The number of hidden units between fc1 and fc2.
        dropout: The dropout rate. Defaults to 0.2.
        fc1: The first fully connected layer after the output.
        fc2: The second fully connected layer after fc1.
        device: The device to run the model on.

    """

    def __init__(
        self,
        marginal_models: list[MarginalFC],
        hidden_dim: int = 128,
        dropout: float = 0.2,
        device: torch.device | None = None,
    ) -> None:
        """Initialize a joint VAE model for integrating multiple views.

        Args:
            marginal_models: The list of marginal models of class FC_Marginal
            hidden_dim: The number of hidden units between fc1 and fc2. Defaults to 128.
            dropout: The dropout rate. Defaults to 0.2.
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

        first_model = marginal_models[0]

        self.fc1 = nn.Linear(first_model.z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, first_model.prediction_dim)
        self.dropout = nn.Dropout(dropout)

        if device is not None:
            self.to(device)

    def forward(self, x: list[torch.Tensor]):
        """FOrward method for joint VAE."""
        assert len(x) == len(self.margin_models), "Number of inputs must match number of marginal models"

        dists = []
        yhats = []

        for i, model in enumerate(self.margin_models):
            if not isinstance(x[i], torch.Tensor):
                continue

            if self.device is not None and x[i].device != self.device:
                x[i] = x[i].to(self.device)

            yhat, dist = model(x[i])

            dists.append(dist)
            yhats.append(yhat)

        # compute the mean and variance of the product of normal distributions
        poe_dist = self.product_of_experts(dists)

        z = poe_dist.rsample() if self.training else poe_dist.loc

        out = self.dropout(tf.relu(self.fc1(z)))
        out = add_last_layer(self.fc2(out), self.prediction_mode)

        return out, poe_dist, yhats, dists

    def loss(
        self,
        y: torch.Tensor,
        yhat: torch.Tensor,
        poe_dist: torch.distributions.Normal,
        yhats: list[torch.Tensor],
        dists: list[torch.distributions.Normal],
        var_beta: float = 1.0,
        focal: bool = True,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the total loss for all the joint and marginal models.

        Args:
            y: Ground truth labels
            yhat: logit predictions for the product of experts
            poe_dist: The distribution of the product of experts
            yhats: List of logit predictions for each marginal model
            dists: List of distributions for each marginal model
            var_beta: The weight of the KL divergence term. Defaults to 1.
            focal: Whether to use focal loss. Defaults to True.
            gamma: The focal loss gamma parameter. Defaults to 2.
            alpha: A tensor with number of elements equal to the number of classes, specifying class weights. Defaults to None.

        Returns:
            torch.Tensor: The joint loss

        """
        # Move data to device if needed
        if self.device is not None:
            if y.device != self.device:
                y = y.to(self.device)
            if yhat.device != self.device:
                yhat = yhat.to(self.device)
            if alpha is not None and alpha.device != self.device:
                alpha = alpha.to(self.device)

        assert isinstance(poe_dist, Normal), "Only Normal distributions are supported"

        product_loss = self.var_loss(y, yhat, poe_dist, var_beta, focal, gamma, alpha)

        # compute the variational loss for each marginal model
        var_losses = [
            self.var_loss(y, yhat, dist, var_beta, focal, gamma, alpha)
            for yhat, dist in zip(yhats, dists, strict=False)
        ]

        return product_loss + sum(var_losses)

    def var_loss(
        self,
        y: torch.Tensor,
        yhat: torch.Tensor,
        dist: torch.distributions.Normal,
        var_beta: float = 1.0,
        focal: bool = True,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the variational loss for a single marginal model.

        Args:
            y: Ground truth labels
            yhat: Prediction probabilities for the labels
            dist: The distribution of the latent variable
            var_beta: The weight of the KL divergence term. Defaults to 1.
            focal: Whether to use focal loss. Defaults to True.
            gamma: The focal loss gamma parameter. Defaults to 2.
            alpha: A tensor with number of elements equal to the number of classes, specifying class weights. Defaults to None.

        Returns:
            torch.Tensor: The variational loss

        """
        assert isinstance(dist, Normal), "Only Normal distributions are supported"

        # Make sure everything is on the right device
        if self.device is not None:
            if y.device != self.device:
                y = y.to(self.device)
            if yhat.device != self.device:
                yhat = yhat.to(self.device)

        # compute the KL divergence between the prior and the posterior
        zeros = torch.zeros_like(dist.mean)
        ones = torch.ones_like(dist.variance)
        if self.device is not None:
            zeros = zeros.to(self.device)
            ones = ones.to(self.device)

        kl = kl_divergence(dist, Normal(zeros, ones)).sum(dim=-1)
        match self.prediction_mode:
            case PredictionMode.CLASSIFICATION:
                # compute the cross entropy loss
                ce = tf.cross_entropy(yhat, y, reduction="none")

                if alpha is not None:
                    if self.device is not None and alpha.device != self.device:
                        alpha = alpha.to(self.device)
                    alpha = alpha.repeat(yhat.shape[0], 1)
                    alpha = alpha.gather(1, y.view(-1, 1))
                    ce = ce * alpha.view(-1)

                if focal:
                    probs = tf.softmax(yhat, dim=1)
                    focal_loss = torch.pow(1 - probs.gather(1, y.view(-1, 1)), gamma).view(-1) * ce
                    return torch.mean(focal_loss + var_beta * kl)
                
                #ce = ce.sum()
                return torch.mean(ce + var_beta * kl)
            
            case PredictionMode.REGRESSION:
                mse = tf.mse_loss(yhat.view(-1), y, reduction="none")
                #mse = mse.sum()
                return torch.mean(mse + var_beta * kl)

    @staticmethod
    def product_of_experts(
        dists: list[Normal],
        eps: float = 1e-8,
    ) -> torch.distributions.Normal:
        """Compute the mean and variance of the product of N normal distributions.

        Args:
            dists: List of torch.distributions.Normal
            eps: Added to division for numerical stability

        Returns:
            torch.distributions.Normal: The product of experts distribution

        """
        # assume mu_0 = 0, S_0 = I
        var_poe = 1
        mu_poe = 0

        for dist in dists:
            assert isinstance(dist, Normal), "Only Normal distributions are supported"
            var_ = dist.variance
            mu_ = dist.mean

            var_poe = var_poe + torch.div(1.0, var_ + eps)
            mu_poe = mu_poe + torch.div(mu_, var_ + eps)

        var_poe = torch.div(1.0, var_poe + eps)
        mu_poe = torch.mul(mu_poe, var_poe)

        return Normal(mu_poe, torch.sqrt(var_poe))

    def freeze_marginal_layers(self) -> None:
        """Freeze marginal layers."""
        for model in self.margin_models:
            freeze_layers(model)

    def freeze_joint_layers(self) -> None:
        """Freeze joint layers."""
        freeze_layers(self, ["fc1"])


def make_joint_vae(
    data_dim: list[int],
    prediction_dim: int,
    hidden_sizes: list[list[int]],
    z_dim: int,
    hidden_dim: int,
    prediction_mode: PredictionMode,
    dropout: float = 0.2,
    device: torch.device | None = None,
) -> JointVAE:
    """Create a joint model for multiple views. Each view gets its own 'marginal model', and then there is a fusion model that takes the output of each marginal model and combines them.

    Args:
        data_dim: A list of input dimensions, each representing a view
        prediction_dim: The number of output classes
        hidden_sizes: A list of hidden sizes for each marginal model
        z_dim: The dimensionality of the latent representation
        hidden_dim: The number of hidden units between fc1 and fc2 of the combination model.
        dropout: The dropout rate
        device: The device to run the model on. Defaults to None.

    Returns:
        JointVAE: The joint model

    """
    marginal_models = []

    for k in range(len(data_dim)):
        input_size = data_dim[k]
        mmod = MarginalFC(
            input_size=input_size,
            hidden_sizes=hidden_sizes[k],
            z_dim=z_dim,
            prediction_dim=prediction_dim,
            prediction_mode=prediction_mode,
            dropout=dropout,
            device=device,
        )
        marginal_models.append(mmod)

    # joint model
    return JointVAE(
        marginal_models=marginal_models,
        hidden_dim=hidden_dim,
        dropout=dropout,
        device=device,
    )
