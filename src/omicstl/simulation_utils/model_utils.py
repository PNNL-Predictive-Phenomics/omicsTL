"""Utilities for training and evaluating transfer learning models."""

import logging
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, Tuple, TypedDict

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, LeaveOneOut, ParameterGrid, StratifiedKFold
from torch import device
import re

from omicstl.simulation_utils.data_utils import DatasetContainer
from omicstl.transfer_forest import TransferForest, PredictionMode, load_r_functions
from omicstl.transfer_networks import TransferMLP, TransferVAE

logger = logging.getLogger(__name__)

# Constants for metric optimization
MINIMIZE_METRICS = ["rmse", "mae"]
DEFAULT_REGRESSION_METRIC = "rmse"
DEFAULT_CLASSIFICATION_METRIC = "acc"
DEFAULT_LOO_CLASSIFICATION_METRIC = "acc"


@dataclass
class EvaluationContext:
    """Context for model evaluation with metadata."""

    scenario: int
    replicate: int
    split_name: str
    model_type: str
    hyperparams: dict[str, Any]


@dataclass
class DataPartition:
    """Container for training data."""

    features: pd.DataFrame
    response: pd.Series
    view_column_groups: dict[str, list[str]] | None = None

    @property
    def views(self) -> list[pd.DataFrame]:
        """Return per-view feature DataFrames.

        Single-omics: returns [self.features].
        Multi-omics: splits self.features by view_column_groups and returns one DataFrame per view.
        """
        if self.view_column_groups is None:
            return [self.features]
        return [self.features[cols] for cols in self.view_column_groups.values()]


@dataclass
class TrainingDict(TypedDict):
    training: DataPartition
    early_stopping: DataPartition | None


@dataclass
class CVFoldData:
    """Container for cross-validation fold data."""

    train: DataPartition
    validation: DataPartition
    early_stopping: DataPartition | None = None
    fold_idx: int = 0

    def get_training_dict(self) -> TrainingDict:
        return {"training": self.train, "early_stopping": self.early_stopping}


@dataclass
class ModelConfig:
    """Configuration for model creation and training."""

    model_type: str
    hyperparams: dict[str, Any]
    is_classification: bool
    torch_device: device
    output_dim: int
    n_views: int = 1


@dataclass
class TuningResult(NamedTuple):
    """Result of hyperparameter tuning."""

    best_params: dict[str, Any]
    results: pd.DataFrame


def create_results_df() -> pd.DataFrame:
    """Create an empty DataFrame with columns for model evaluation results."""
    return pd.DataFrame(
        columns=[
            "scenario",
            "replicate",
            "split",
            "model",
            "model_type",
            "model_id",
            "rmse",
            "mae",
            "acc",
            "f1",
            "mcc",
            "precision",
            "recall",
        ],
    )


def calculate_metrics(
    true_values: pd.Series | np.ndarray,
    predicted_values: pd.Series | np.ndarray,
    is_classification: bool,
    predicted_probs: np.ndarray | None = None
) -> dict[str, float]:
    """Calculate evaluation metrics for model predictions.

    Args:
    true_values: Ground truth values
    predicted_values: Model predictions (class labels for classification)
    predicted_probs: Model predicted probabilities (must be specified only for classification)
    is_classifiction: Is the model a classification model

    Returns:
    Dictionary containing calculated metrics

    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        matthews_corrcoef,
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics: dict[str, float] = {}

    # Convert to numpy arrays if they're pandas Series
    if isinstance(true_values, pd.Series):
        true_values = true_values.to_numpy()
    if isinstance(predicted_values, pd.Series):
        predicted_values = predicted_values.to_numpy()

    metrics["rmse"] = np.nan
    metrics["mae"] = np.nan
    metrics["acc"] = np.nan
    metrics["roc_auc"] = np.nan
    metrics["mcc"] = np.nan
    metrics["f1"] = np.nan
    metrics["precision"] = np.nan
    metrics["recall"] = np.nan
        
    if is_classification:
        unique_classes = np.unique(true_values)
        
        truth_classes = true_values
        predicted_classes = predicted_values

        # Remap binary classification from 1, 2 to 0, 1
        if len(unique_classes) == 2 and np.min(truth_classes) == 1:
            truth_classes = truth_classes - 1
            predicted_classes = predicted_classes - 1

        metrics["acc"] = float(accuracy_score(y_true = truth_classes, 
                                              y_pred = predicted_classes))
        if predicted_probs is not None:
            metrics["roc_auc"] = float(roc_auc_score(y_true = truth_classes, 
                                                    y_score = predicted_probs,
                                                    multi_class = "ovr" if len(unique_classes) > 2 else "raise",
                                                    average = 'weighted'))
        metrics["mcc"] = float(matthews_corrcoef(y_true = truth_classes, 
                                                 y_pred = predicted_classes))
        metrics["f1"] = float(f1_score(y_true = truth_classes, 
                                       y_pred = predicted_classes, 
                                       average = "weighted" if len(unique_classes) > 2 else "binary"))
        metrics["precision"] = float(precision_score(y_true = truth_classes, 
                                                     y_pred = predicted_classes, 
                                                     average = "weighted" if len(unique_classes) > 2 else "binary",
                                                     zero_division=0))
        metrics["recall"] = float(recall_score(y_true = truth_classes, 
                                               y_pred = predicted_classes, 
                                               average = "weighted" if len(unique_classes) > 2 else "binary",
                                               zero_division=0))

        return metrics

    metrics["rmse"] = float(np.sqrt(mean_squared_error(true_values, predicted_values)))
    metrics["mae"] = float(mean_absolute_error(true_values, predicted_values))

    return metrics


def create_model(config: ModelConfig) -> TransferMLP | TransferVAE:
    """Create a model instance based on configuration.

    Args:
    config: Model configuration

    Returns:
    Initialized model instance

    """
    logger.debug("Creating model. Formating parameters:")
    dropout = config.hyperparams.get("dropout", 0.2)
    hidden_dim_base = int(config.hyperparams.get("hidden_dim_base", 12))
    z_dim_base = int(config.hyperparams.get("z_dim_base", 12))
    n_latent_dims = int(config.hyperparams.get("n_latent_dims", 2))
    hidden_sizes = [hidden_dim_base * 2 * (i + 1) for i in range(int(n_latent_dims) - 1)]
    hidden_sizes.reverse()
    logger.debug("dropout: %s", dropout)
    logger.debug("hidden_dim_base: %s", hidden_dim_base)
    logger.debug("z_dim_base: %s", z_dim_base)
    logger.debug("n_latent_dims: %s", n_latent_dims)
    logger.debug("hidden_sizes: %s", hidden_sizes)

    if config.model_type == "mult_mlp":
        model = TransferMLP(
            hidden_sizes=[hidden_sizes] * config.n_views,
            dropout=dropout,
            hidden_dim=hidden_dim_base,
        )
    elif config.model_type == "mult_vae":
        model = TransferVAE(
            hidden_sizes=[hidden_sizes] * config.n_views,
            dropout=dropout,
            hidden_dim=hidden_dim_base,
            z_dim=z_dim_base,
            var_beta=config.hyperparams.get("var_beta", 0.01),
        )
    else:
        logger.error("Unknown model type: %s", config.model_type)
        raise ValueError

    if config.is_classification:
        model.with_classification()
    else:
        model.with_regression()

    return model


def convert_parameter_types(params: dict[str, Any]) -> dict[str, Any]:
    """Convert parameters to their correct types.

    Args:
    params: Dictionary of parameters that may have NumPy types

    Returns:
    Dictionary with parameters converted to Python native types

    """
    param_types = {
        "hidden_dim_size": int,
        "source_epochs": int,
        "target_epochs": int,
        "z_dim": int,
        "dropout": float,
    }

    converted_params = {}
    for key, value in params.items():
        if key in param_types:
            converted_params[key] = param_types[key](value)
        else:
            converted_params[key] = value

    return converted_params


def evaluate_model(
    model: TransferMLP | TransferVAE,
    data: DataPartition,
    model_id: str,
    context: EvaluationContext,
    is_classification: bool,
) -> pd.DataFrame:
    """Evaluate a model and create a results row.

    Args:
    model: Model to evaluate
    data: Data for evaluation
    model_id: ID to use when generating predictions ("source" or "target")
    context: Evaluation context with metadata
    is_classification: is the model a classification model

    Returns:
    DataFrame with a single row containing evaluation results

    """
    predictions = model.__getattribute__(model_id).predict(data.views)

    if is_classification:
        predictions = pd.Series(predictions)
        predictions_probs = model.__getattribute__(model_id).predict(data.views, return_probabilities = True)
        if predictions_probs.shape[1] == 2:
            predictions_probs = predictions_probs[:, 1]

        # using training instead of validation to capture unique 
        # values since it should capture all possible values of response, whereas validation set may be unbalanced and omit classes
        # This code is assuming that the categorical responses are coded
        # beginning from 1, and not from 0. So a binary response would be
        # (1,2), not (0,1). Similarly a tertiary response would be (1,2,3),
        # not (0,1,2). If the native coding for categorical values in the
        # data objects of omicstl is changed, this code should be changed
        # too. 
        response_unique_sorted = sorted(data.response.unique()) 
        preds_unique_sorted = sorted(predictions.unique()) 
        sort_diff = [a - b for a, b in zip(response_unique_sorted, preds_unique_sorted)]
        sort_diff_val = set(sort_diff).pop().item()
        if sort_diff_val != 0:
            # mapping = {val - 1: val for val in response_unique_sorted}
            mapping = {val - sort_diff_val: val for val in response_unique_sorted}
            predictions = predictions.map(mapping)
        metrics = calculate_metrics(true_values = data.response, 
                                    predicted_values = predictions, 
                                    is_classification = is_classification,
                                    predicted_probs = predictions_probs)
    else:
        metrics = calculate_metrics(true_values = data.response, 
                                    predicted_values = predictions, 
                                    is_classification = is_classification)

    results_row = {
        "scenario": context.scenario,
        "replicate": context.replicate,
        "split": context.split_name,
        "model": context.model_type,
        "model_type": context.model_type,
        "model_id": model_id,
        **metrics,
        **context.hyperparams,
    }

    return pd.DataFrame([results_row])


def train_model(
    model: TransferMLP | TransferVAE,
    data: TrainingDict,
    config: ModelConfig,
    model_id: str,
    source_model: str | None = None,
    lr: float = 0.001,
    gamma: float = 2,
    weight_decay: float = 1e-4
) -> None:
    """Train a model on the provided data.

    Args:
        model: Model to train
        data: Training data
        config: Model configuration
        model_id: ID to use for the model ("source" or "target")
        source_model: Name of source model for transfer learning
        lr: learning rate
    """
    train_data = data["training"]
    early_stopping_data = data["early_stopping"]
    # Force dims update when initialising the target model — its feature set may
    # differ from the source model that was trained first on this same object.
    force_dims = source_model is not None
    try:
        model.set_model_dims(train_data.views, config.output_dim, force=force_dims)
    except Exception:
        logger.exception("Error setting model dimensions")

    if source_model:
        model.create_model(model_id, source_model=source_model, device=config.torch_device, lr=lr)

        freeze_mode = config.hyperparams.get("freeze", "none")
        if model_id == "target" and freeze_mode in ("marginal", "encoders") and hasattr(model, "target"):
            target_container = getattr(model, "target")
            target_container.model.freeze_marginal_layers()
    else:
        model.create_model(model_id, device=config.torch_device, lr=lr, weight_decay=weight_decay)

    epochs_key = "target_epochs" if "target" in model_id else "source_epochs"
    epochs = config.hyperparams.get(epochs_key, 1000)

    model.train_model(
        model_id,
        train_views=train_data.views,
        y_train=train_data.response,
        epochs=epochs,
        verbose=True,
        test_views=early_stopping_data.views if early_stopping_data else None,
        y_test=early_stopping_data.response if early_stopping_data else None,
        gamma=float(config.hyperparams.get("gamma", gamma)),
    )

def predict_dl_model(
    pretrained_model: TransferMLP | TransferVAE,
    features: list[pd.DataFrame] | pd.DataFrame,
    model_id: str = "target",
    return_probabilities = False
):
    views = features if isinstance(features, list) else [features]
    return pretrained_model.__getattribute__(model_id).predict(views, return_probabilities=return_probabilities)

def create_cv_folds(
    feature_matrix: pd.DataFrame,
    response_values: pd.Series,
    n_splits: int,
) -> list[CVFoldData]:
    """Create cross-validation folds from the data.

    Args:
    feature_matrix: Feature data
    response_values: Response data
    n_splits: Number of CV folds (will be adjusted for small datasets)

    Returns:
    List of CV fold data

    """
    is_classification = response_values.dtype == np.int64

    if is_classification:
        class_counts = response_values.value_counts()
        min_class_count = class_counts.min()
        if min_class_count < n_splits:
            logger.info("Insufficient samples for stratification, using Leave-One-Out cross-validation")
            cv_splitter = LeaveOneOut()
        else:
            logger.info("Using stratified %s-fold cross-validation", n_splits)
            cv_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        logger.info("Using regular %s-fold cross-validation", n_splits)
        cv_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(feature_matrix, response_values)):
        train_features = feature_matrix.iloc[train_idx]
        val_features = feature_matrix.iloc[val_idx]
        train_response = response_values.iloc[train_idx]
        val_response = response_values.iloc[val_idx]

        fold_data = CVFoldData(
            train=DataPartition(features=train_features, response=train_response),
            validation=DataPartition(features=val_features, response=val_response),
            fold_idx=fold_idx,
        )

        cv_folds.append(fold_data)

    if not cv_folds:
        logger.warning("No valid CV folds could be created, using all data for both training and validation")
        fold_data = CVFoldData(
            train=DataPartition(features=feature_matrix, response=response_values),
            validation=DataPartition(features=feature_matrix, response=response_values),
            fold_idx=0,
        )
        cv_folds.append(fold_data)

    logger.info("Created %s cross-validation folds", len(cv_folds))
    return cv_folds


def aggregate_fold_metrics(fold_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metrics across folds.

    Args:
        fold_metrics: List of metric dictionaries from each fold

    Returns:
        Dictionary with averaged metrics

    """
    if not fold_metrics:
        return {}

    fold_df = pd.DataFrame(fold_metrics)

    numeric_cols = fold_df.select_dtypes(include=np.number).columns.tolist()
    non_numeric_cols = [col for col in fold_df.columns if col not in numeric_cols and col != "fold"]

    mean_metrics: dict[str, Any] = {
        str(k): v for k, v in fold_df[numeric_cols].mean().to_dict().items()
    }

    for col in non_numeric_cols:
        mean_metrics[col] = fold_df[col].iloc[0]

    return mean_metrics


def select_best_parameters(
    results: pd.DataFrame,
    param_grid: dict[str, list[Any]],
    metric: str,
) -> dict[str, Any]:
    """Select the best parameters based on the specified metric.

    Args:
    results: DataFrame with aggregated metrics
    param_grid: Parameter grid used for tuning
    metric: Metric to optimize

    Returns:
    Dictionary with best parameters

    """
    if results.empty:
        return {}

    param_cols = list(param_grid.keys())

    if metric in results.columns and not results[metric].isna().all():
        best_idx = results[metric].idxmin() if metric in MINIMIZE_METRICS else results[metric].idxmax()
        best_val = results[metric].min() if metric in MINIMIZE_METRICS else results[metric].max()
        best_params = {k: results.loc[best_idx, k] for k in param_cols}
        logger.info(f"Best parameter combination found. Metric {metric} with value {best_val}")
        logger.info(best_params)
        logger.info(results[metric])
    else:
        logger.warning("All %s values are NaN, using first parameter combination", metric)
        if len(results) > 0:
            best_params = {k: results.iloc[0][k] for k in param_cols}
        else:
            return {}

    return convert_parameter_types(best_params)


def _create_loo_folds(
    feature_matrix: pd.DataFrame,
    response_values: pd.Series,
    view_column_groups: dict[str, list[str]] | None = None,
) -> list[CVFoldData]:
    cv_splitter = LeaveOneOut()
    all_folds = list(cv_splitter.split(feature_matrix, response_values))

    cv_folds = []
    for val_fold_idx, (train_indices, val_indices) in enumerate(all_folds):
        np.random.seed(42 + val_fold_idx)
        early_stop_indices = np.random.choice(train_indices, size=1, replace=False)

        train_indices = np.array([idx for idx in train_indices if idx not in early_stop_indices])

        train_features = feature_matrix.iloc[train_indices]
        train_response = response_values.iloc[train_indices]

        val_features = feature_matrix.iloc[val_indices]
        val_response = response_values.iloc[val_indices]

        early_stop_features = feature_matrix.iloc[early_stop_indices]
        early_stop_response = response_values.iloc[early_stop_indices]

        fold_data = CVFoldData(
            train=DataPartition(features=train_features, response=train_response, view_column_groups=view_column_groups),
            validation=DataPartition(features=val_features, response=val_response, view_column_groups=view_column_groups),
            early_stopping=DataPartition(features=early_stop_features, response=early_stop_response, view_column_groups=view_column_groups),
            fold_idx=val_fold_idx,
        )

        cv_folds.append(fold_data)

    return cv_folds


def _create_kfold_folds(
    feature_matrix: pd.DataFrame,
    response_values: pd.Series,
    cv_splitter: KFold | StratifiedKFold,
    view_column_groups: dict[str, list[str]] | None = None,
) -> list[CVFoldData]:
    all_folds = list(cv_splitter.split(feature_matrix, response_values))
    n_actual_splits = len(all_folds)

    cv_folds = []
    for val_fold_idx in range(n_actual_splits):
        _, val_indices = all_folds[val_fold_idx]

        early_stop_fold_idx = (val_fold_idx + 1) % n_actual_splits
        _, early_stop_indices = all_folds[early_stop_fold_idx]

        train_indices = np.concatenate(
            [
                all_folds[i][1]
                for i in range(n_actual_splits)
                if i != val_fold_idx and i != early_stop_fold_idx
            ]
        ).astype(np.int_, copy=False)

        train_features = feature_matrix.iloc[train_indices]
        train_response = response_values.iloc[train_indices]

        val_features = feature_matrix.iloc[val_indices]
        val_response = response_values.iloc[val_indices]

        early_stop_features = feature_matrix.iloc[early_stop_indices]
        early_stop_response = response_values.iloc[early_stop_indices]

        fold_data = CVFoldData(
            train=DataPartition(features=train_features, response=train_response, view_column_groups=view_column_groups),
            validation=DataPartition(features=val_features, response=val_response, view_column_groups=view_column_groups),
            early_stopping=DataPartition(features=early_stop_features, response=early_stop_response, view_column_groups=view_column_groups),
            fold_idx=val_fold_idx,
        )

        cv_folds.append(fold_data)

    return cv_folds


def create_k_minus_2_cv_folds(
    feature_matrix: pd.DataFrame,
    response_values: pd.Series,
    n_splits: int,
    view_column_groups: dict[str, list[str]] | None = None,
) -> list[CVFoldData]:
    if n_splits < 3:
        logger.warning("n_splits must be at least 3 for k-2 fold CV, setting to 3")
        n_splits = 3

    is_classification = response_values.dtype == np.int64
    n_samples = feature_matrix.shape[0]

    if n_samples < n_splits:
        cv_folds = _create_loo_folds(feature_matrix, response_values, view_column_groups)
    elif is_classification:
        class_counts = response_values.value_counts()
        min_class_count = class_counts.min()
        if min_class_count < n_splits:
            logger.info("Insufficient samples for stratification, using Leave-One-Out cross-validation")
            cv_folds = _create_loo_folds(feature_matrix, response_values, view_column_groups)
        else:
            logger.info("Using stratified %s-fold cross-validation for k-2 CV", n_splits)
            cv_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_folds = _create_kfold_folds(feature_matrix, response_values, cv_splitter, view_column_groups)
    else:
        logger.info("Using regular %s-fold cross-validation for k-2 CV", n_splits)
        cv_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_folds = _create_kfold_folds(feature_matrix, response_values, cv_splitter, view_column_groups)

    if not cv_folds:
        # TODO: Maybe there is a better alternative exception
        logger.warning("No valid CV folds could be created, using all data for all partitions")
        fold_data = CVFoldData(
            train=DataPartition(features=feature_matrix, response=response_values, view_column_groups=view_column_groups),
            validation=DataPartition(features=feature_matrix, response=response_values, view_column_groups=view_column_groups),
            early_stopping=DataPartition(features=feature_matrix, response=response_values, view_column_groups=view_column_groups),
            fold_idx=0,
        )
        cv_folds.append(fold_data)

    logger.info("Created %s cross-validation folds using k-2 split strategy", len(cv_folds))
    return cv_folds


def tune_model_cv(
    data_partition: DataPartition,
    model_type: Literal["mult_mlp", "mult_vae"],
    output_dim: int,
    torch_device: device,
    param_grid: dict[str, Any],
    n_splits: int = 5,
    metric: str | None = None,
) -> dict[str, Any]:
    """Tune model hyperparameters using cross-validation on the provided data partition.

    Args:
        data_partition: Data partition to use for tuning
        model_type: Type of model to tune
        torch_device: Torch device to train model using
        param_grid: Dictionary of parameter names and values to search
        n_splits: Number of CV folds (will be adjusted for small datasets)
        metric: Performance metric to optimize

    Returns:
        Dictionary of best parameters

    """
    is_classification = data_partition.response.dtype == np.int64

    if is_classification and len(np.unique(data_partition.response)) <= 1:
        logger.warning("Only one class present in the entire dataset, metrics will be undefined")
        return {}

    n_views = len(data_partition.view_column_groups) if data_partition.view_column_groups else 1
    cv_folds = create_k_minus_2_cv_folds(
        data_partition.features, data_partition.response, n_splits,
        view_column_groups=data_partition.view_column_groups,
    )

    using_loo = any(len(fold.validation.features) == 1 for fold in cv_folds)

    if metric is None:
        if is_classification:
            metric = DEFAULT_LOO_CLASSIFICATION_METRIC if using_loo else DEFAULT_CLASSIFICATION_METRIC
        else:
            metric = DEFAULT_REGRESSION_METRIC

    logger.info("Using %s as optimization metric", metric)

    results = pd.DataFrame()
    param_combinations = list(ParameterGrid(param_grid))
    logger.info("Tuning over %s parameter combinations", len(param_combinations))

    for param_idx, hyperparams in enumerate(param_combinations):
        logger.debug("Evaluating parameter combination %s/%s", param_idx + 1, len(param_combinations))
        fold_metrics = []

        for fold_data in cv_folds:
            config = ModelConfig(
                model_type=model_type,
                hyperparams=hyperparams,
                is_classification=is_classification,
                torch_device=torch_device,
                output_dim=output_dim,
                n_views=n_views,
            )

            model = create_model(config)
            train_model(
                model=model,
                data=fold_data.get_training_dict(),
                config=config,
                model_id="source",
                lr=config.hyperparams.get("lr", 0.01),
                weight_decay=hyperparams.get("weight_decay", 0.01),
            )

            predictions = model.source.predict(fold_data.validation.views)
            predictions = pd.Series(predictions)

            if is_classification:
                predictions_probs = model.source.predict(fold_data.validation.views, return_probabilities = True)
                if predictions_probs.shape[1] == 2:
                    predictions_probs = predictions_probs[:, 1]

                # using training instead of validation to capture unique 
                # values since it should capture all possible values of response, whereas validation set may be unbalanced and omit classes
                # This code is assuming that the categorical responses are coded
                # beginning from 1, and not from 0. So a binary response would be
                # (1,2), not (0,1). Similarly a tertiary response would be (1,2,3),
                # not (0,1,2). If the native coding for categorical values in the
                # data objects of omicstl is changed, this code should be changed
                # too. 
                response_unique_sorted = sorted(fold_data.train.response.unique())
                preds_unique_sorted = sorted(predictions.unique())
                sort_diff = [a - b for a, b in zip(response_unique_sorted, preds_unique_sorted)]
                sort_diff_val = set(sort_diff).pop().item()
                if sort_diff_val != 0:
                    mapping = {val - sort_diff_val: val for val in response_unique_sorted}
                    predictions = predictions.map(mapping) 
                metrics = calculate_metrics(true_values = fold_data.validation.response, 
                                            predicted_values = predictions, 
                                            is_classification = is_classification,
                                            predicted_probs = predictions_probs)
            else:
                metrics = calculate_metrics(true_values = fold_data.validation.response, 
                                            predicted_values = predictions, 
                                            is_classification = is_classification)

            metrics.update(hyperparams)
            metrics["fold"] = fold_data.fold_idx

            fold_metrics.append(metrics)

        if fold_metrics:
            mean_metrics = aggregate_fold_metrics(fold_metrics)
            results = pd.concat([results, pd.DataFrame([mean_metrics])], ignore_index=True)

    if results.empty:
        logger.warning("No successful parameter combinations found, returning defaults")
        return {}

    best_params = select_best_parameters(results, param_grid, metric)
    logger.info("Best parameters found: %s", best_params)
    return best_params


def get_feature_columns(data: pd.DataFrame, response_id: str) -> list[str]:
    """Extract feature column names from dataset.

    Args:
        data: DataFrame containing features and response
        response_id: String identifier for response column in data.

    Returns:
        List of feature column names

    """
    return [col for col in data.columns if col != response_id]


def split_dataframe_indices(
    df: pd.DataFrame,
    first_split_proportion: float = 0.8,
    random_state: int | None = None,
) -> tuple[list[int], list[int]]:
    """Split row indices of a pandas DataFrame into two sets based on a proportion.

    Args:
        df: The pandas DataFrame to split indices from.
        first_split_proportion: Proportion of indices to include in the first set (0.0-1.0).
        random_state: Optional random seed for reproducibility.

    Returns:
        A tuple containing two lists of indices: (first_indices, second_indices).

    """
    if not 0 <= first_split_proportion <= 1:
        msg = "first_split_proportion must be between 0.0 and 1.0"
        raise ValueError(msg)

    all_indices = df.index.tolist()

    indices = np.array(all_indices)

    if random_state is not None:
        np.random.seed(random_state)

    np.random.shuffle(indices)

    split_idx = int(len(indices) * first_split_proportion)

    if len(indices) > 0:
        split_idx = max(1, min(split_idx, len(indices) - 1))

    first_indices = indices[:split_idx].tolist()
    second_indices = indices[split_idx:].tolist()

    return first_indices, second_indices


def create_data_partition(
        data: pd.DataFrame,
        feature_cols: list[str],
        response_id: str,
        row_ids: list | None = None,
        view_column_groups: dict[str, list[str]] | None = None,
) -> DataPartition:
    """Create a data partition from a DataFrame.

    Args:
        data: DataFrame containing features and response
        feature_cols: List of feature column names
        row_ids: Optional list of row indices to subset the data. If None, uses all data.
        view_column_groups: Optional mapping of view name to column list for multi-omics.
            If None, all feature columns are treated as a single view.

    Returns:
        DataPartition object

    """
    subset_data = data.loc[row_ids] if row_ids else data

    return DataPartition(
        features=subset_data[feature_cols],
        response=subset_data[response_id],
        view_column_groups=view_column_groups,
    )


def fit_dl_model(
    data_container: DatasetContainer,
    model_type: Literal["mult_mlp", "mult_vae"],
    torch_device: device,
    param_grid: dict[str, float | bool | str] | None = None,
    mmd_weight: float | None = None,
    target_lr: float | None = None,
    **kwargs: float | str,
) -> tuple[pd.DataFrame, TransferMLP | TransferVAE | None, TransferMLP | TransferVAE | None]:
    """Fit a deep learning model with optional hyperparameter tuning via cross-validation.

    Args:
        data_container: Container with source and target data
        model_type: Type of model to fit ("mult_mlp" or "mult_vae")
        torch_device: Torch device for deep learning
        param_grid: Dictionary of parameter grids for tuning
        **kwargs: Parameters for the model if not tuning

    Returns:
        DataFrame with model evaluation results

    """
    results = create_results_df()

    if not data_container.id_tuple:
        msg = "data_container contains null id_tuple"
        raise AttributeError(msg)
    replicate, scenario = data_container.id_tuple

    response_id = data_container.response_id
    is_classification = data_container.is_classification()
    view_column_groups = data_container.view_column_groups
    n_views = len(view_column_groups) if view_column_groups else 1

    source_feature_cols = get_feature_columns(data_container.source_data, response_id=response_id)
    target_feature_cols = get_feature_columns(data_container.target_data, response_id=response_id)

    # Create source partitions (always use source feature columns)
    source_train_samples, source_validation_samples = split_dataframe_indices(data_container.source_data, 0.9)

    source_train_partition = create_data_partition(
        data=data_container.source_data,
        feature_cols=source_feature_cols,
        response_id=response_id,
        row_ids=source_train_samples,
        view_column_groups=view_column_groups,
    )

    source_validation_partition = create_data_partition(
        data=data_container.source_data,
        feature_cols=source_feature_cols,
        response_id=response_id,
        row_ids=source_validation_samples,
        view_column_groups=view_column_groups,
    )

    # Create target partitions (always use target feature columns)
    target_train_samples, target_validation_samples = split_dataframe_indices(data_container.target_data, 0.9)

    target_full = create_data_partition(
        data=data_container.target_data,
        feature_cols=target_feature_cols,
        response_id=response_id,
        view_column_groups=view_column_groups,
    )

    target_train_partition = create_data_partition(
        data=data_container.target_data,
        feature_cols=target_feature_cols,
        response_id=response_id,
        row_ids=target_train_samples,
        view_column_groups=view_column_groups,
    )

    target_validation_partition = create_data_partition(
        data=data_container.target_data,
        feature_cols=target_feature_cols,
        response_id=response_id,
        row_ids=target_validation_samples,
        view_column_groups=view_column_groups,
    )

    source_partition: TrainingDict = {"training": source_train_partition, "early_stopping": source_validation_partition}
    target_partition: TrainingDict = {"training": target_train_partition, "early_stopping": target_validation_partition}
    target_partition_nosource: TrainingDict = {"training": target_train_partition, "early_stopping": target_validation_partition}

    output_dim = data_container.source_data[response_id].nunique() if is_classification else 1

    best_params_nosource: dict = {}

    if param_grid is not None:
        # target_epochs is not tunable via source CV — strip it so it does not
        # appear to be searched when it is actually ignored during source training.
        source_param_grid = {k: v for k, v in param_grid.items() if k != "target_epochs"}
        logger.info("Tuning parameters using source data")
        best_params = tune_model_cv(
            data_partition=source_train_partition,
            model_type=model_type,
            output_dim=output_dim,
            torch_device=torch_device,
            param_grid=source_param_grid,
        )

        if best_params:
            kwargs.update(best_params)
            logger.debug("Source hyperparameter tuning complete with parameters: %s", best_params)

        logger.info("Tuning parameters using target data")
        best_params_nosource = tune_model_cv(
            data_partition=target_train_partition,
            model_type=model_type,
            output_dim=output_dim,
            torch_device=torch_device,
            param_grid=param_grid,
        )

        if best_params_nosource:
            logger.debug("Target hyperparameter tuning complete with parameters: %s", best_params_nosource)

    # Source + target model
    logger.debug("Training transfer model")

    _source_lr = float(kwargs.get("lr", 1e-4))
    hyperparams = {
        "dropout": kwargs.get("dropout", 0.2),
        "hidden_dim_base": kwargs.get("hidden_dim_base", 12),
        "n_latent_dims": kwargs.get("n_latent_dims", 2),
        "source_epochs": kwargs.get("source_epochs", 1000),
        "target_epochs": kwargs.get("target_epochs", 1000),
        "freeze": kwargs.get("freeze", "none"),
        "z_dim_base": kwargs.get("z_dim_base", 12),
        "lr": _source_lr,
        # Use full lr for target fine-tuning by default. lr/10 prevents catastrophic
        # forgetting when source and target are close, but causes stalling when the
        # domain shift is large (separate PCA spaces) and source pre-training has
        # already pushed weights into a source-specific basin.
        "target_lr": target_lr if target_lr is not None else _source_lr,
        "weight_decay": kwargs.get("weight_decay", 1e-4),
        "gamma": kwargs.get("gamma", 2),
        "var_beta": kwargs.get("var_beta", 0.01),
    }
    logger.debug("Using hyperparameters: %s", hyperparams)

    config = ModelConfig(
        model_type=model_type,
        hyperparams=hyperparams,
        is_classification=is_classification,
        torch_device=torch_device,
        output_dim=output_dim,
        n_views=n_views,
    )
    model = create_model(config)

    logger.info("Creating and training model on source domain")
    train_model(
        model=model,
        data=source_partition,
        config=config,
        model_id="source",
        lr=float(hyperparams.get("lr", 0.01)),
        gamma=float(hyperparams.get("gamma", 2)),
        weight_decay=float(hyperparams.get("weight_decay", 0.01)),
    )

    if not hasattr(data_container, "target_data") or data_container.target_data is None:
        logger.warning("No target data found, returning early")
        return results, None, None

    logger.info("Transferring to target domain")
    train_model(
        model=model,
        data=target_partition,
        config=config,
        model_id="target",
        source_model="source",
        lr=float(hyperparams["target_lr"]),
        gamma=float(hyperparams.get("gamma", 2)),
        weight_decay=float(hyperparams.get("weight_decay", 0.01))
    )

    # Target only model
    logger.debug("Training transfer model")

    hyperparams_nosource = {
        "dropout": best_params_nosource.get("dropout", 0.2),
        "hidden_dim_base": best_params_nosource.get("hidden_dim_base", 12),
        "n_latent_dims": best_params_nosource.get("n_latent_dims", 2),
        "source_epochs": best_params_nosource.get("source_epochs", 1000),
        # train_model picks epochs via "target_epochs" when model_id contains "target"
        # (which it does for "target_nosource"). Map the CV-tuned source_epochs here
        # so the no-source model actually uses the epoch count found by CV.
        "target_epochs": best_params_nosource.get("source_epochs", 1000),
        "freeze": best_params_nosource.get("freeze", "none"),
        "z_dim_base": best_params_nosource.get("z_dim_base", 12),
        "lr": best_params_nosource.get("lr", 1e-4),
        "weight_decay": best_params_nosource.get("weight_decay", 1e-4),
        "gamma": best_params_nosource.get("gamma", 2),
    }
    logger.debug("Using hyperparameters: %s", hyperparams_nosource)

    config_nosource = ModelConfig(
        model_type=model_type,
        hyperparams=hyperparams_nosource,
        is_classification=is_classification,
        torch_device=torch_device,
        output_dim=output_dim,
        n_views=n_views,
    )
    model_nosource = create_model(config_nosource)
    train_model(
        model=model_nosource,
        data=target_partition_nosource,
        config=config_nosource,
        model_id="target_nosource",
        lr=best_params_nosource.get("lr", 0.01),
        gamma=best_params_nosource.get("gamma", 2),
        weight_decay=best_params_nosource.get("weight_decay", 0.01),
    )

    if not hasattr(data_container, "target_test_data") or not data_container.target_test_data:
        logger.warning("No test data found, returning early")
        return results, None, None

    for i, test_dataset in enumerate(data_container.target_test_data):
        logger.info("Evaluating on test set %s", i + 1)
        test_data = create_data_partition(
            data=test_dataset,
            response_id=response_id,
            feature_cols=target_feature_cols,
            view_column_groups=view_column_groups,
        )

        test_context = EvaluationContext(
            scenario=scenario,
            replicate=replicate,
            split_name=f"test_{i}",
            model_type=model_type,
            hyperparams=hyperparams,
        )
        test_results = evaluate_model(model=model, data=test_data, model_id="target", context=test_context, is_classification=is_classification)
        logger.info(f"Transfer results: {test_results}")
        results = pd.concat([results, test_results], ignore_index=True)

        test_context = EvaluationContext(
            scenario=scenario,
            replicate=replicate,
            split_name=f"test_{i}",
            model_type=model_type,
            hyperparams=hyperparams_nosource,
        )
        test_results = evaluate_model(
            model=model_nosource,
            data=test_data,
            model_id="target_nosource",
            context=test_context,
            is_classification=is_classification,
        )
        logger.info(f"No transfer results: {test_results}")
        results = pd.concat([results, test_results], ignore_index=True)

    return results, model, model_nosource

def reorg_rf_predictions(test_predictions) -> dict:
    # Grouping definition and new keys
    groupings = {
        'truth': ['truth'],
        'pred_source_full': [r'^pred_source(?!.*val).*'],
        'pred_source_full_val': [r'^pred_source.*val.*'],
        'pred_0_full': [r'^pred_0(?!.*val).*'],
        'pred_0_full_val': [r'^pred_0.*val.*'],
        'pred_1_full': [r'^pred_1(?!.*val).*'],
        'pred_1_full_val': [r'^pred_1.*val.*'],
        'pred_2_full': [r'^pred_2(?!.*val).*'],
        'pred_2_full_val': [r'^pred_2.*val.*'],
        'pred_3_full': [r'^pred_3(?!.*val).*'],
        'pred_3_full_val': [r'^pred_3.*val.*'],
        'pred_ensemble_full': [r'^pred_ensemble(?!.*val).*'],
        'pred_ensemble_full_val': [r'^pred_ensemble.*val.*']
    }

    # Initialize the new dictionary where values will be reorganized
    test_predictions_reorg = {}

    # Iterate over each group, apply regex filters, and build dataframes
    for new_key, patterns in groupings.items():
        matched_keys = []
        for pattern in patterns:
            # Use regex to match keys in the dictionary
            matched_keys.extend([key for key in test_predictions.keys() if re.match(pattern, key)])
        
        if matched_keys:
            # Create a pandas DataFrame by column-binding arrays corresponding to matched keys
            df = pd.DataFrame({key: test_predictions[key] for key in matched_keys})
            test_predictions_reorg[new_key] = df

    return test_predictions_reorg

def fit_rf_model(
    data_container: DatasetContainer
) -> Tuple[pd.DataFrame, TransferForest]:
    """Fit a random forest transfer learning model.

    Args:
        data_container: Container with source and target data.

    Returns:
        DataFrame with model evaluation results and the fitted TransferForest.

    """
    load_r_functions()

    logger.info("Starting random forest model fitting")
    results = create_results_df()

    if data_container.id_tuple is None:
        raise ValueError("data_container does not have any loaded datasets")
    
    replicate, scenario = data_container.id_tuple
    response_id = data_container.response_id
    logger.info(f"Processing scenario: {scenario}, replicate: {replicate}")

    is_classification = data_container.is_classification()
    logger.info(f"Task type: {'Classification' if is_classification else 'Regression'}")

    view_column_groups = data_container.view_column_groups

    source_feature_cols = get_feature_columns(data=data_container.source_data, response_id=response_id)
    target_feature_cols = get_feature_columns(data=data_container.target_data, response_id=response_id)

    logger.info(f"Using {len(source_feature_cols)} source / {len(target_feature_cols)} target features")

    source_data = create_data_partition(
        data=data_container.source_data,
        response_id=response_id,
        feature_cols=source_feature_cols,
        view_column_groups=view_column_groups,
    )
    target_data = create_data_partition(
        data=data_container.target_data,
        response_id=response_id,
        feature_cols=target_feature_cols,
        view_column_groups=view_column_groups,
    )

    logger.info("Initializing TransferForest model")
    transfer_forest = TransferForest()
    if is_classification:
        transfer_forest.with_classification()
        logger.info("Set model for classification task")
    else:
        transfer_forest.with_regression()
        logger.info("Set model for regression task")

    logger.info("Training model on source data")
    transfer_forest.train_models(
        views=source_data.views,
        response=source_data.response,
    )

    logger.info("Updating model with target data")
    transfer_forest.update_models(
        views=target_data.views,
        response=target_data.response,
    )

    # Target-only RF — analogous to target_nosource for DL models.
    # Trained on target data only, no source pretraining.
    logger.info("Initializing target-only TransferForest model")
    target_only_forest = TransferForest()
    if is_classification:
        target_only_forest.with_classification()
    else:
        target_only_forest.with_regression()
    target_only_forest.train_models(
        views=target_data.views,
        response=target_data.response,
    )
    # update_models must be called so generate_predictions finds non-empty transfer_models
    target_only_forest.update_models(
        views=target_data.views,
        response=target_data.response,
    )

    if hasattr(data_container, "target_test_data") and data_container.target_test_data:
        logger.info(f"Found {len(data_container.target_test_data)} test datasets")
        for i, test_dataset in enumerate(data_container.target_test_data):
            logger.info(f"Processing test dataset {i + 1}/{len(data_container.target_test_data)}")

            test_data = create_data_partition(
                data=test_dataset,
                response_id=response_id,
                feature_cols=target_feature_cols,
                view_column_groups=view_column_groups,
            )

            ensemble_views = None
            ensemble_response = None
            if data_container.target_ensemble_data is not None:
                ensemble_data = create_data_partition(
                    data_container.target_ensemble_data,
                    response_id=response_id,
                    feature_cols=target_feature_cols,
                    view_column_groups=view_column_groups,
                )

                ensemble_views = ensemble_data.views
                ensemble_response = ensemble_data.response

            test_predictions = transfer_forest.generate_predictions(
                views=test_data.views,
                response=test_data.response,
                validation_views=test_data.views,
                validation_response=test_data.response,
                ensemble_views=ensemble_views,
                ensemble_response=ensemble_response,
                integration_type=TransferForest.IntegrationType.NONE,
            )

            if not test_predictions:
                logger.warning(f"Empty predictions returned for test data {i}")

            test_predictions_reorg = reorg_rf_predictions(test_predictions[0])

            for model_type, prediction in test_predictions_reorg.items():
                logger.info(f"Calculating metrics for test data {i} with model type: {model_type}")
                if str.endswith(model_type, "truth"):
                    continue
                
                if is_classification:
                    model_type_base = model_type.replace("_full", "")
                    predicted_classes = prediction[model_type_base].to_numpy().astype(int)

                    predicted_probs = prediction.drop(columns = model_type_base)
                    if predicted_probs.shape[1] == 2:
                        predicted_probs = predicted_probs.iloc[:,1].to_numpy()
                    else:
                        predicted_probs = predicted_probs.to_numpy()

                    test_metrics = calculate_metrics(true_values = test_data.response, 
                                                     predicted_values = predicted_classes, 
                                                     is_classification = is_classification,
                                                     predicted_probs = predicted_probs)

                else:
                    model_type_base = model_type.replace("_full", "")
                    predicted_values = prediction[model_type_base].to_numpy()
                    test_metrics = calculate_metrics(true_values = test_data.response, 
                                                     predicted_values = predicted_values, 
                                                     is_classification = is_classification)

            
                test_row = {
                    "scenario": scenario,
                    "replicate": replicate,
                    "split": f"test_{i}",
                    "model": "rf",
                    "model_type": f"{model_type}",
                    "model_id": "target",
                    **test_metrics,
                }
                results = pd.concat([results, pd.DataFrame([test_row])], ignore_index=True)

            # ---- Evaluate target-only forest on same test split
            target_only_predictions = target_only_forest.generate_predictions(
                views=test_data.views,
                response=test_data.response,
                validation_views=test_data.views,
                validation_response=test_data.response,
                ensemble_views=None,
                ensemble_response=None,
                integration_type=TransferForest.IntegrationType.NONE,
            )

            if not target_only_predictions:
                logger.warning(f"Empty predictions from target-only RF for test data {i}")
            else:
                target_only_reorg = reorg_rf_predictions(target_only_predictions[0])

                # Only the pred_source_full group exists (no transfer step was run).
                # Relabel it as pred_target_full to distinguish it from the
                # source-only predictions produced by the main transfer_forest.
                pred_to = target_only_reorg.get("pred_source_full")
                if pred_to is not None:
                    if is_classification:
                        predicted_classes = pred_to["pred_source"].to_numpy().astype(int)
                        predicted_probs = pred_to.drop(columns="pred_source")
                        if predicted_probs.shape[1] == 2:
                            predicted_probs = predicted_probs.iloc[:, 1].to_numpy()
                        else:
                            predicted_probs = predicted_probs.to_numpy()
                        target_only_metrics = calculate_metrics(
                            true_values=test_data.response,
                            predicted_values=predicted_classes,
                            is_classification=is_classification,
                            predicted_probs=predicted_probs,
                        )
                    else:
                        predicted_values = pred_to["pred_source"].to_numpy()
                        target_only_metrics = calculate_metrics(
                            true_values=test_data.response,
                            predicted_values=predicted_values,
                            is_classification=is_classification,
                        )

                    target_only_row = {
                        "scenario": scenario,
                        "replicate": replicate,
                        "split": f"test_{i}",
                        "model": "rf",
                        "model_type": "pred_target_full",
                        "model_id": "target_nosource",
                        **target_only_metrics,
                    }
                    results = pd.concat([results, pd.DataFrame([target_only_row])], ignore_index=True)

    logger.info(f"Model fitting completed. Results shape: {results.shape}")
    return results, transfer_forest

def update_rf_model(
    pretrained_model : TransferForest,
    response_id : str,
    target_data : pd.DataFrame,
    target_test_data : list[pd.DataFrame] | None,
    target_ensemble_data : pd.DataFrame | None,
    view_column_groups: dict[str, list[str]] | None = None,
) -> Tuple[pd.DataFrame, TransferForest]:
    is_classification = pretrained_model._prediction_mode == PredictionMode.CLASSIFICATION
    logger.info(f"Task type: {'Classification' if is_classification else 'Regression'}")

    feature_cols = get_feature_columns(
        data=target_data,
        response_id=response_id,
    )
    logger.info(f"Using {len(feature_cols)} features for modeling")

    target_data_update = create_data_partition(
        data=target_data,
        response_id=response_id,
        feature_cols=feature_cols,
        view_column_groups=view_column_groups,
    )

    transfer_model = pretrained_model.copy()

    transfer_model.update_models(target_data_update.views, target_data_update.response)

    if target_test_data is not None:
        logger.info(f"Found {len(target_test_data)} test datasets")
        for i, test_dataset in enumerate(target_test_data):
            logger.info(f"Processing test dataset {i + 1}/{len(target_test_data)}")

            test_data = create_data_partition(
                data=test_dataset,
                response_id=response_id,
                feature_cols=feature_cols,
                view_column_groups=view_column_groups,
            )

            logger.info(f"Test data {i} shape: {test_data.features.shape}")

            ensemble_views = None
            ensemble_response = None
            if target_ensemble_data is not None:
                ensemble_data = create_data_partition(
                    target_ensemble_data,
                    response_id=response_id,
                    feature_cols=feature_cols,
                    view_column_groups=view_column_groups,
                )

                ensemble_views = ensemble_data.views
                ensemble_response = ensemble_data.response

            test_predictions = transfer_model.generate_predictions(
                views=test_data.views,
                response=test_data.response,
                validation_views=test_data.views,
                validation_response=test_data.response,
                ensemble_views=ensemble_views,
                ensemble_response=ensemble_response,
                integration_type=TransferForest.IntegrationType.NONE,
            )

            if not test_predictions:
                logger.warning(f"Empty predictions returned for test data {i}")

            # print("CALCULATE")
            test_predictions = test_predictions[0]
            for model_type, prediction in test_predictions.items():
                logger.info(f"Calculating metrics for test data {i} with model type: {model_type}")
                if str.endswith(model_type, "_prob"):
                    continue
                
                test_metrics = calculate_metrics(test_data.response, prediction, is_classification)
                test_row = {
                    "split": f"test_{i}",
                    "model": "rf",
                    "model_type": f"{model_type}",
                    "model_id": "target",
                    **test_metrics,
                }
                results = pd.concat([results, pd.DataFrame([test_row])], ignore_index=True)

    logger.info(f"Model fitting completed. Results shape: {results.shape}")
    return results, transfer_model

def predict_rf_model(
    pretrained_model: TransferForest,
    input_data: pd.DataFrame,
) -> dict:
    """Predict response values using a pre-trained random forest transfer model.

    Args:
        pretrained_model: the trained TransferForest model to use to predict.
    """
    return pretrained_model.generate_predictions([input_data])[0]
