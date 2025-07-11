"""API for fitting deep learning models in a transfer learning context."""
import sys

sys.path.append("../")


import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as tf
from torch import nn
from torch.optim import AdamW
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from omicstl.deep_learning_utils import PredictionMode, compute_accuracy, compute_mse
from omicstl.mult_mlp import make_joint_model
from omicstl.mult_vae import make_joint_vae

logger = logging.getLogger(__name__)


class MultiViewModel:
    """A general container for deep learning based models for multi-omics applications."""

    def __init__(
        self,
    ) -> None:
        """Initialize a container for deep learning based TL models."""
        self.source_model: MultiViewModel.ModelObject
        self.target_model: MultiViewModel.ModelObject
        self._view_dims: list[float]

    class ModelObject:
        """A single instance of a deep learning model."""

        def __init__(
            self,
            model: nn.Module,
            optimizer: Optimizer,
            scheduler: LRScheduler,
            prediction_mode: PredictionMode,
        ) -> None:
            self.model = model
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.loss_history: list[float] = []
            self.train_metric: list[float] = []
            self.test_metric: list[float] = []
            self.prediction_mode = prediction_mode

        def predict(self, views: list[pd.DataFrame]) -> np.ndarray:
            """Make predictions using the model based on input views.

            Args:
                    views: List of pandas DataFrames containing the input data for each view

            Returns:
                    np.ndarray: Predictions in the format determined by prediction_mode

            Raises:
                    ValueError: Unknown prediction mode

            """
            tensor_views = self.as_tensors(views)

            self.model.eval()

            with torch.inference_mode():
                yhat, *_ = self.model(tensor_views)

                match self.prediction_mode:
                    case PredictionMode.REGRESSION:
                        predictions = yhat.view(-1).detach().cpu().numpy()
                    case PredictionMode.CLASSIFICATION:
                        predictions = yhat.argmax(dim=1).detach().cpu().numpy()
                    case _:
                        raise ValueError(f"Unknown prediction mode: {self.prediction_mode}")

            return predictions

        def train(
            self,
            train_views: list[pd.DataFrame],
            y_train: pd.DataFrame,
            loss_fn: Callable,
            epochs: int,
            test_views: list[pd.DataFrame] | None = None,
            y_test: pd.DataFrame | None = None,
            verbose: bool = False,
            patience: int = 50,
            min_delta: float = 1e-4,
            gamma: float = 2,
        ) -> None:
            """Train the model with early stopping support."""
            train_metric = []
            test_metric = []
            loss_history = []

            train_tensors: list[torch.Tensor] = self.as_tensors(train_views)
            y_train_tensor: torch.Tensor = self._parse_response(y_train)

            if y_test is not None:
                y_test_tensor: torch.Tensor = self._parse_response(y_test)
            if test_views is not None:
                test_tensors: list[torch.Tensor] = self.as_tensors(test_views)

            best_metric = float("inf") if self.prediction_mode == PredictionMode.REGRESSION else -float("inf")
            epochs_without_improvement = 0

            for i in range(epochs):
                loss, yhat = loss_fn(self, train_tensors, y_train_tensor, gamma = gamma)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                self.optimizer.step()

                match self.prediction_mode:
                    case PredictionMode.REGRESSION:
                        train_metric.append(compute_mse(predictions=yhat.view(-1), targets=y_train_tensor))
                    case PredictionMode.CLASSIFICATION:
                        train_metric.append(compute_accuracy(predictions=yhat, targets=y_train_tensor))

                loss_history.append(loss.item())

                logger.debug("Epoch %d loss: %.3f", i + 1, loss.item())

                if y_test is None or test_views is None:
                    continue

                with torch.inference_mode():
                    yhat_test, _, _, _ = self.model(test_tensors)

                current_metric = None
                improved = False

                match self.prediction_mode:
                    case PredictionMode.REGRESSION:
                        current_metric = compute_mse(predictions=yhat_test.view(-1), targets=y_test_tensor)
                        test_metric.append(current_metric)
                        if best_metric - current_metric > min_delta:
                            improved = True
                    case PredictionMode.CLASSIFICATION:
                        current_metric = compute_accuracy(predictions=yhat_test, targets=y_test_tensor)
                        test_metric.append(current_metric)
                        if current_metric - best_metric > min_delta:
                            improved = True

                self.scheduler.step(current_metric)

                if improved:
                    best_metric = current_metric
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        if verbose:
                            logger.info(f"Early stopping at epoch {i + 1} — no improvement in {patience} epochs.")
                        break

            self.loss_history = loss_history
            self.train_metric = train_metric
            self.test_metric = test_metric

        def _parse_response(self, response: pd.DataFrame) -> torch.Tensor:
            """Format response in appropriate way depending on the models prediction type."""
            match self.prediction_mode:
                case PredictionMode.CLASSIFICATION:
                    new_response = response.astype("category").cat.codes
                    new_response = torch.tensor(new_response.values, dtype=torch.int64)
                case PredictionMode.REGRESSION:
                    new_response = torch.tensor(response.values, dtype=torch.float32)

            return new_response

        def as_tensors(self, views: list[pd.DataFrame]) -> list[torch.Tensor]:
            tensor_views = []
            for view_df in views:
                tensor = torch.tensor(view_df.values, dtype=torch.float32)
                tensor_views.append(tensor)
            return tensor_views

    def with_classification(self) -> "MultiViewModel":
        """Puts the model in classification mode. Should be called before any training."""
        self._prediction_mode = PredictionMode.CLASSIFICATION
        return self

    def with_regression(self) -> "MultiViewModel":
        """Puts the model in regression mode. Should be called before any training."""
        self._prediction_mode = PredictionMode.REGRESSION
        return self

    def set_model_dims(self, views: list[pd.DataFrame], output_dim: int) -> None:
        view_dims = []
        output_dim: int

        for k in range(len(views)):
            view_dims.append(views[k].shape[1])
        if not hasattr(self, "_view_dims"):
            self._view_dims = view_dims

        if not self._prediction_mode:
            print("Set prediction mode before creating model")
            raise

        if not hasattr(self, "_output_dims"):
            self._output_dim = output_dim

    def train_model(
        self,
        model_id: str,
        train_views: list[pd.DataFrame],
        y_train: pd.Series,
        epochs: int,
        test_views: list[pd.DataFrame] | None = None,
        y_test: pd.Series | None = None,
        verbose: bool = False,
		gamma: float = 2,
    ):
        """Trains the specified model dynamically.

        Args:
                model_id: The attribute name of the model to train.
                train_views: A list of data matrices (multi-omics views).
                y_train: The response variable dataframe.
                epochs: Number of epochs for training.
                test_views: A list of data matrices (multi-omics views).
                y_test: The response variable dataframe.
                verbose: If true, enables verbose printing per epoch.

        Raises:
                                        AttributeError: model_id does not exist

        """
        if hasattr(self, model_id):
            model = getattr(self, model_id)
        else:
            raise AttributeError(f"Model '{model_id}' does not exist in the object.")

        model.train(
            train_views=train_views,
            y_train=y_train,
            loss_fn=self._compute_loss,
            epochs=epochs,
            test_views=test_views,
            y_test=y_test,
            verbose=verbose,
            gamma=gamma,
        )

    def _compute_loss(self, model_object: ModelObject, views: list, response: torch.Tensor):
        """An abstract method which should be overridden in each model.
        Wraps the model loss to improve readability and reduce code duplication for training loop.
        """
        print("_compute_loss() not overridden for inherited MultiViewModel")


class TransferVAE(MultiViewModel):
    """A general container for a variational autoencoder (VAE) model used in multi-omics transfer learning."""

    def __init__(
        self,
        hidden_sizes: list[list[int]],
        z_dim: int,
        dropout: float,
        hidden_dim: int,
    ) -> None:
        """Initializes a container for a variational autoencoder (VAE) model used in multi-omics transfer learning.

        Args:
                hidden_sizes: A list of hidden sizes for each marginal model
                z_dim: The dimensionality of the latent representation.
                dropout: The dropout rate. Defaults to 0.2.
                hidden_dim: The number of hidden units between fc1 and fc2 of the combination model. #TODO: improve this docstring

        """
        self.hidden_sizes = hidden_sizes
        self.z_dim = z_dim
        self.dropout = dropout
        self.hidden_dim = hidden_dim

    def create_model(
        self,
        model_id: str,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        source_model: str | None = None,
        device=None,
    ) -> None:
        if hasattr(self, "_view_dims"):
            view_dims = self._view_dims
            output_dim = self._output_dim
        else:
            raise AttributeError("view_dims not found, run set_model_dims first")

        model = make_joint_vae(
            data_dim=view_dims,
            prediction_dim=output_dim,
            hidden_sizes=self.hidden_sizes,
            z_dim=self.z_dim,
            dropout=self.dropout,
            hidden_dim=self.hidden_dim,
            prediction_mode=self._prediction_mode,
            device=device,
        )

        if source_model is not None:
            if not hasattr(self, source_model):
                raise AttributeError(f"Model '{source_model}' does not exist in the object")
            source_model_object = getattr(self, source_model)
            model.load_state_dict(source_model_object.model.state_dict())

        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        mode = "min" if self._prediction_mode == PredictionMode.REGRESSION else "max"
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, patience=10, factor=0.2)
        model_object = self.ModelObject(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            prediction_mode=self._prediction_mode,
        )
        setattr(self, model_id, model_object)

    def _compute_loss(
        self,
        model_object: MultiViewModel.ModelObject,
        views: list,
        response: torch.Tensor,
        gamma: float = 2,
    ):
        """Wraps the model loss to improve readability and reduce code duplication for training loop."""
        yhat, poe_dist, yhats, dists = model_object.model(views)
        loss = model_object.model.loss(response, yhat, poe_dist, yhats, dists, var_beta=0.01, gamma = gamma)
        return loss, yhat


class TransferMLP(MultiViewModel):
    """A general container for a multilayer perceptron (MLP) model used in multi-omics transfer learning."""

    def __init__(
        self,
        hidden_sizes: list[list[int]],
        dropout: float,
        hidden_dim: int,
        activation_fn: Callable = tf.relu,
        combine_fn: str = "concat",  # TODO: make this an enum, probably in utils
        device=None,
    ) -> None:
        """Initializes a container for a MLP model used in multi-omics transfer learning.

        Args:
                hidden_sizes: A list of hidden sizes for each marginal model
                dropout: The dropout rate. Defaults to 0.2.
                hidden_dim: The number of hidden units between fc1 and fc2 of the combination model. #TODO: improve this docstring
                activation_fn: A callable representing the activation functions of each layer in the MLP model. Typically from torch.nn.functional
                combine_fn: A string specifying the integration strategy of the model

        """
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.activation_fn = activation_fn
        self.combine_fn = combine_fn

    def create_model(
        self,
        model_id: str,
        lr: float = 1e-4,
        weight_decay = 1e-4,
        source_model: str | None = None,
        device=None,
    ) -> None:
        if hasattr(self, "_view_dims"):
            view_dims = self._view_dims
            output_dim = self._output_dim
        else:
            raise AttributeError("view_dims not found, run set_model_dims first")

        model = make_joint_model(
            data_dim=view_dims,
            prediction_dim=output_dim,
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout,
            hidden_dim=self.hidden_dim,
            prediction_mode=self._prediction_mode,
            activation_fn=self.activation_fn,
            combine_fn=self.combine_fn,
            device=device,
        )

        if source_model is not None:
            if not hasattr(self, source_model):
                raise AttributeError(f"Model '{source_model}' does not exist in the object")
            source_model_object = getattr(self, source_model)
            model.load_state_dict(source_model_object.model.state_dict())

        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        mode = "min" if self._prediction_mode == PredictionMode.REGRESSION else "max"
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, patience=10, factor=0.2)

        model_object = self.ModelObject(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            prediction_mode=self._prediction_mode,
        )
        setattr(self, model_id, model_object)

    def _compute_loss(
        self,
        model_object: MultiViewModel.ModelObject,
        views: list,
        response: torch.Tensor,
        gamma: float = 2,
    ):
        """Wraps the model loss to improve readability and reduce code duplication for training loop."""
        yhat, _, yhats, _ = model_object.model(views)
        _, _, loss = model_object.model.loss(response, yhat, yhats, gamma = gamma)
        return loss, yhat
