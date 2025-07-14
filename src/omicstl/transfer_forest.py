from enum import Enum, auto
from typing import Any

import logging
import numpy as np
import pandas as pd
from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FactorVector, FloatVector
from sklearn.linear_model import LinearRegression

from omicstl.r_utils import df2pd, pd2df
from omicstl.deep_learning_utils import PredictionMode
from omicstl._defs import _PKG_ROOT

logger = logging.getLogger(__name__)

def load_r_functions(path: str | None = None) -> None:
    """Initialize R environment and loads transfer learning random forest logic from R.

    This method sources the R script containing transfer forest functionality.
    """
    if path is None:
        path = _PKG_ROOT()

    logger.info("Initializing R environment")
    base = importr("base")
    base.Sys_setenv(OMICSTL_PKG_ROOT = _PKG_ROOT())
    base.source(file=f"{path}/r/transfer_forest.R")
    logger.info("R environment initialized")

class TransferForest:
    """A container for random forest based multi-omics transfer learning objects and methods."""

    def __init__(self) -> None:
        """Initialize a container for random forest based multi-omics transfer learning objects and methods."""
        self.source_models: list = []
        self.transfer_models: list = []

    class ModelType(Enum):
        """An enum for specifying a transfer model. Used to elimate dependancy on strings."""

        IMPORTANCE_WEIGHTED = auto()
        """Ordinary random forest model"""

        LEARNED_DEPENDENCY = auto()
        """Model 1 as specified in *reference here*"""

        APPENDED_DEPENDENCY = auto()
        """Model 2 as specified in *reference here*"""

        BASE = auto()
        """Model 3 as specified in *reference here*"""

        ENSEMBLE = auto()
        """Ensemble model as specified in *reference here*"""

    class UpdateType(Enum):
        """Type of transfer learning to apply."""

        SOURCE = auto()
        """Update on the source model"""

        ITERATE = auto()
        """Update on the most recent TL model"""

    class IntegrationType(Enum):
        """An enum for determining how to update the pre-trained models. Used to elimate dependancy on strings."""

        NONE = auto()
        """View specific predictions are provided."""

        BAYESIAN = auto()
        """Uses a Bayesian approach for view integration."""

        WEIGHTED = auto()
        """Uses a weighted approach for view integration, with weights calculated using a linear model."""

    def with_classification(self) -> "TransferForest":
        """Set the prediction mode to classification for binary or multi-class outcomes.

        Returns:
            TransferForest: The TransferForest instance for method chaining.

        Raises:
            RuntimeError: If source models are already trained, preventing mode change.

        """
        if self.source_models:
            logger.info("Source models already trained, cannot change prediction mode.")
            return self
        self._prediction_mode = PredictionMode.CLASSIFICATION
        return self

    def with_regression(self) -> "TransferForest":
        """Set the prediction mode to regression for continuous outcomes.

        Returns:
            TransferForest: The TransferForest instance for method chaining.

        Raises:
            RuntimeError: If source models are already trained, preventing mode change.

        """
        if self.source_models:
            logger.info("Source models already trained, cannot change prediction mode.")
            return self
        self._prediction_mode = PredictionMode.REGRESSION
        return self

    def train_models(
        self,
        views: list[pd.DataFrame],
        response: pd.DataFrame,
    ) -> None:
        """Trains an initial random forest model using each source view.

        Args:
            views: A list of k = 1, ..., K, n x p_k data matrices for the available multi-omics "source" views.
            response: The length n response list associated with the provided views.

        """
        if not self._prediction_mode:
            msg = "Specify a prediction mode before calling train_models. See with_classification or with_regression."
            logger.info(msg)
            # TODO: Raise ValueError instead of returning
            return

        for index, view in enumerate(views):
            try:
                model = robjects.r["viRandomForests"](
                    y=self._resolve_response(response),
                    x=pd2df(view),
                    importance=True,
                )
                self.source_models.append(model)
            except Exception as ex:
                logger.info("Fitting model for view %s failed with error %s.", index, ex)

    def update_models(
        self,
        views: list[pd.DataFrame],
        response: pd.DataFrame,
        model_type: ModelType | None = None,
        update_type: UpdateType = UpdateType.SOURCE,
        override: bool = False,
    ) -> None:
        """Update the transfer models using target data for transfer learning.

        Args:
            views: A list of k = 1, ..., K, n x p_k data matrices for the available multi-omics "target" views. Source and target dimensions must match.
            response: The length n response list associated with the provided views.
            model_type: The type of transfer model to use (IMPORTANCE_WEIGHTED, LEARNED_DEPENDENCY, APPENDED_DEPENDENCY, etc.).
            update_type: The model update strategy. SOURCE uses original source models as reference, ITERATE uses previously trained transfer models.
            override: Whether to override existing transfer models if they exist.

        Raises:
            ValueError: If source models have not been trained yet.

        """
        if self.source_models is None:
            logger.info("Source models have not been trained yet. Run train_models first.")
            # TODO: Raise ValueError instead of returning
            return

        if update_type == self.UpdateType.ITERATE and not self.transfer_models:
            logger.info("Souce models have not been updated with new data yet. Defaulting to source model update.")
            update_type = self.UpdateType.SOURCE

        if update_type == self.UpdateType.SOURCE:
            if self.transfer_models and not override:
                logger.info("Updated models already exist! If you want to override them, run with override = True")
                return
            reference_models = self.source_models

        else:
            reference_models = self._get_transfer_models(model_type)

        if not reference_models:
            logger.info("Source model not found.")
            return

        for index, view in enumerate(views):
            reference_model = reference_models[index]
            source_importance = robjects.r["importancenew"](reference_model, 2)
            self.transfer_models.append(
                robjects.r["train_trans_rf"](
                    x=pd2df(view),
                    y=self._resolve_response(response),
                    rf_source=reference_model,
                    var_importance_source=source_importance,
                ),
            )

    def generate_predictions(
        self,
        views: list[pd.DataFrame],
        response: pd.DataFrame,
        validation_views: list[pd.DataFrame],  # = None
        validation_response: pd.DataFrame,  # = None
        integration_type: IntegrationType = IntegrationType.NONE,
    ) -> list[dict]:
        """Predicts using the transfer models on new data with optional integration of predictions across views.

        Args:
            views: A list of k = 1, ..., K, n x p_k data matrices for the target views to generate predictions for.
            response: The response dataframe associated with the target views (may be used for integration).
            validation_views: A list of validation data matrices used for validating predictions or training integration models.
            validation_response: The response dataframe associated with the validation views.
            integration_type: The method used to integrate predictions across views (NONE, BAYESIAN, or WEIGHTED).

        Returns:
            list[dict]: A list of dictionaries containing predictions for each view and model. Format varies based on integration_type.
            When integration_type is NONE, returns a list of dictionaries, one per view.
            When integration_type is BAYESIAN or WEIGHTED, returns a single-element list with one dictionary.

        Raises:
            ValueError: If transfer models have not been trained yet.

        """
        logger.info("Generating predictions")
        if not self.transfer_models:
            logger.info("Transfer models have not been trained yet. Run update_models first.")
            # TODO: Raise ValueError instead of returning empty dict
            return [{}]

        prediction_dicts = []
        validation_dicts = []
        for index, view in enumerate(views):
            view_transfer_models = self.transfer_models[index]
            res = robjects.r["predict_trans_rf"](
                trans_rf_res=view_transfer_models,
                newdata=pd2df(view),
                x_val=pd2df(validation_views[index]),
                y_val=self._resolve_response(validation_response),
            )
            prediction_dicts.append(df2pd(res).to_dict(orient="list"))

            if integration_type != self.IntegrationType.WEIGHTED:
                continue

            res = robjects.r["predict_trans_rf"](
                trans_rf_res=view_transfer_models,
                newdata=pd2df(validation_views[index]),
                x_val=pd2df(validation_views[index]),
                y_val=self._resolve_response(validation_response),
            )
            validation_dicts.append(df2pd(res).to_dict(orient="list"))

        match integration_type:
            case self.IntegrationType.NONE:
                return prediction_dicts
            case self.IntegrationType.BAYESIAN:
                if self._prediction_mode == PredictionMode.REGRESSION:
                    logger.info("Bayesian Integration unsupported for regression models.")
                    # TODO: spit error here instead of returning
                    return [{}]
                return self._bayesian_integration(prediction_dicts=prediction_dicts, response=response)
            case self.IntegrationType.WEIGHTED:
                # TODO: add checks for validation data if we make that optional
                return self._weighted_integration(
                    prediction_dicts=prediction_dicts,
                    validation_dicts=validation_dicts,
                    validation_response=validation_response,
                )

    def _get_transfer_models(self, model_type: ModelType | None = None) -> list[Any]:
        """Retrieve transfer models of the specified type.

        Args:
            model_type: The type of transfer model to retrieve.

        Returns:
            list[Any]: A list of transfer models of the specified type.

        Raises:
            ValueError: If the specified model type is not found.

        """
        match model_type:
            case self.ModelType.IMPORTANCE_WEIGHTED:
                target_model = "m1"
            case self.ModelType.LEARNED_DEPENDENCY:
                target_model = "m2"
            case self.ModelType.APPENDED_DEPENDENCY:
                target_model = "m3"

        def _get_transfer_model(transfer_models, target_model):
            for name, value in transfer_models.items():
                if name == target_model:
                    return value

        models = []
        for view_transfer_models in self.transfer_models:
            models.append(
                _get_transfer_model(
                    transfer_models=view_transfer_models,
                    target_model=target_model,
                ),
            )
        return models

    def _resolve_response(self, response) -> FactorVector | FloatVector:
        """Convert the response data to the appropriate R vector type based on prediction mode.

        Args:
            response: The response data to convert.

        Returns:
            FactorVector | FloatVector: The converted response as an R vector.

        Raises:
            ValueError: If the prediction mode is not set.

        """
        match self._prediction_mode:
            case PredictionMode.REGRESSION:
                response = FloatVector(response)
            case PredictionMode.CLASSIFICATION:
                response = FactorVector(FloatVector(response), levels=FloatVector(np.unique(response)))
        return response

    def _bayesian_integration(self, prediction_dicts: dict, response) -> list[dict]:
        """Integrates predictions across views using a Bayesian approach.

        Args:
            prediction_dicts: A list of dictionaries containing predictions for each view.
            response: The response data used for calculating class priors.

        Returns:
            list[dict]: A single-element list containing a dictionary with integrated predictions.

        """
        inverted_prediction_dicts = self._invert_prediction_dicts(prediction_dicts=prediction_dicts)

        for model, model_predictions in inverted_prediction_dicts.items():
            combined_predicted_probabilies = np.prod(model_predictions, axis=0).tolist()
            _, class_counts = np.unique(response, return_counts=True)
            class_1_probs = np.array(combined_predicted_probabilies) * (class_counts[-1] / np.sum(class_counts))
            class_0_probs = (1 - np.array(combined_predicted_probabilies)) * (class_counts[1] / np.sum(class_counts))
            inverted_prediction_dicts[model] = class_1_probs / (class_0_probs + class_1_probs)

        return [inverted_prediction_dicts]

    def _weighted_integration(self, prediction_dicts: dict, validation_dicts: dict, validation_response) -> list[dict]:
        """Integrates predictions across views using a weighted approach with linear regression.

        Args:
            prediction_dicts: A list of dictionaries containing predictions for each view.
            validation_dicts: A list of dictionaries containing predictions for validation data.
            validation_response: The true response values for validation data.

        Returns:
            list[dict]: A single-element list containing a dictionary with integrated predictions.

        """
        inverted_validation_dicts = self._invert_prediction_dicts(prediction_dicts=validation_dicts)
        inverted_prediction_dicts = self._invert_prediction_dicts(prediction_dicts=prediction_dicts)

        for model in inverted_validation_dicts:
            predictors = pd.DataFrame()
            test_df = pd.DataFrame()
            for view_index, (val_result, test_result) in enumerate(
                zip(inverted_validation_dicts[model], inverted_prediction_dicts[model], strict=False),
            ):
                predictors[view_index] = val_result
                test_df[view_index] = test_result

            meta_model = LinearRegression()
            meta_model.fit(predictors, validation_response)

            predicted_values = meta_model.predict(test_df)
            inverted_prediction_dicts[model] = predicted_values

        return [inverted_prediction_dicts]

    def _invert_prediction_dicts(self, prediction_dicts: dict) -> dict[str, list]:
        """Inverts the structure of prediction dictionaries to organize by model rather than by view.

        Args:
            prediction_dicts: A list of dictionaries containing predictions for each view.

        Returns:
            dict[str, list]: A dictionary mapping model names to lists of predictions across views.

        """
        models = prediction_dicts[0].keys()
        return {model: [preds[model] for preds in prediction_dicts] for model in models}
