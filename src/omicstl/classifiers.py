"""Thin wrappers around TIMED transfer-learning models for external integration.

Each class exposes three methods:
  save(path)       – persist the trained model to disk
  load(path)       – class method; returns a ready-to-use classifier
  predict(X)       – classify or score samples; returns a list of y-values

The wrappers are intentionally generic: they work for any response type
(binary classification, multi-class, or continuous regression) and any
label encoding, not just the viral/bacterial use case.

Usage
-----
After training in the pipeline::

    from omicstl.classifiers import TIMEDClassifierMLP, TIMEDClassifierRF

    # Classification: pass the sorted unique response values as `classes`
    mlp_clf = TIMEDClassifierMLP(mlp_model, classes=sorted(target_truth.unique()))
    mlp_clf.save("mlp_model.pth")

    # Regression: omit `classes` (or pass None)
    mlp_reg = TIMEDClassifierMLP(mlp_model)
    mlp_reg.save("mlp_reg.pth")

    # RF: classification vs regression is inferred from the model automatically
    rf_clf = TIMEDClassifierRF(rf_model)
    rf_clf.save("rf_model.pkl")

In an external application (e.g. BacterAI)::

    mlp_clf = TIMEDClassifierMLP.load("mlp_model.pth")
    labels  = mlp_clf.predict(new_data)   # original label values, e.g. [1, 2, 1, ...]

    rf_clf  = TIMEDClassifierRF.load("rf_model.pkl")
    labels  = rf_clf.predict(new_data)    # same scale as training response
"""

from __future__ import annotations

import pickle
import re
from typing import Any

import numpy as np
import pandas as pd
import torch

from omicstl.deep_learning_utils import PredictionMode
from omicstl.transfer_forest import TransferForest
from omicstl.transfer_networks import TransferMLP, TransferVAE


class TIMEDClassifierMLP:
    """Classifier / regressor wrapper for a trained TransferMLP or TransferVAE model.

    The wrapper always uses the transfer-learned model (trained on both source
    and target data) rather than the target-only baseline.

    Parameters
    ----------
    model:
        Trained TransferMLP or TransferVAE object returned by ``fit_dl_model``.
    classes:
        For **classification** — the sorted list of original response values,
        e.g. ``[1, 2]`` for a binary task where 1=bacterial and 2=viral, or
        ``[0, 1, 2]`` for a three-class task.  The model's ``argmax`` output
        index ``i`` is mapped back to ``classes[i]``.
        For **regression** — pass ``None`` (default); ``predict`` returns raw
        float predictions.
    """

    def __init__(
        self,
        model: TransferMLP | TransferVAE,
        classes: list[Any] | None = None,
    ) -> None:
        self._model = model
        self._classes = classes

    def predict(self, X: pd.DataFrame) -> list[Any]:
        """Return predictions for every row of *X*.

        Args:
            X: Feature DataFrame with the same columns used during training.

        Returns:
            For classification: list of original label values (one per row),
            in the same space as the training response (e.g. ``[1, 2, 1, ...]``).
            For regression: list of floats.
        """
        raw = self._model.target.predict([X])

        if self._classes is not None:
            # argmax returns 0-indexed class positions; map back to original labels.
            return [self._classes[int(v)] for v in raw]

        return [float(v) for v in raw]

    def save(self, path: str) -> None:
        """Persist the model and its label metadata to disk.

        Args:
            path: Destination file path (recommended extension: ``.pth``).
        """
        torch.save({"model": self._model, "classes": self._classes}, path)

    @classmethod
    def load(cls, path: str) -> TIMEDClassifierMLP:
        """Load a saved model from disk.

        Args:
            path: Path to a ``.pth`` file previously written by :meth:`save`.

        Returns:
            A ready-to-use :class:`TIMEDClassifierMLP` instance.
        """
        data = torch.load(path, weights_only=False)
        return cls(data["model"], data.get("classes"))


class TIMEDClassifierRF:
    """Classifier / regressor wrapper for a trained TransferForest (random forest) model.

    Whether the task is classification or regression is inferred automatically
    from the model's internal prediction mode — no extra arguments required.

    Predictions use the ensemble sub-model (``pred_ensemble_full``) when
    available, falling back to the last available transfer sub-model otherwise.
    """

    def __init__(self, model: TransferForest) -> None:
        self._model = model
        self._is_classification = (
            model._prediction_mode == PredictionMode.CLASSIFICATION
        )

    def predict(self, X: pd.DataFrame) -> list[Any]:
        """Return predictions for every row of *X*.

        Args:
            X: Feature DataFrame with the same columns used during training.

        Returns:
            For classification: list of original integer label values
            (e.g. ``[1, 2, 1, ...]`` matching the training response encoding).
            For regression: list of floats.
        """
        preds_dict = self._model.generate_predictions([X])[0]

        # Try ensemble keys first (with and without the _full suffix).
        key = next(
            (k for k in ("pred_ensemble_full", "pred_ensemble") if k in preds_dict),
            None,
        )
        if key is None:
            # Ensemble weights may not survive serialization.  Fall back to the
            # last individual transfer sub-model CLASS prediction key.
            # Class-prediction keys match r'^pred_\d+$' (e.g. pred_0, pred_3).
            # Probability columns (pred_0_1, pred_0_2 …) are excluded because
            # their float values truncate to 0 under int(), giving wrong results.
            transfer_keys = sorted(
                k for k in preds_dict if re.match(r"^pred_\d+$", k)
            )
            key = transfer_keys[-1] if transfer_keys else next(iter(preds_dict))

        values = preds_dict[key]

        if self._is_classification:
            return [int(v) for v in values]

        return [float(v) for v in values]

    def save(self, path: str) -> None:
        """Persist the model to disk.

        Args:
            path: Destination file path (recommended extension: ``.pkl``).
        """
        with open(path, "wb") as f:
            pickle.dump(self._model, f)

    @classmethod
    def load(cls, path: str) -> TIMEDClassifierRF:
        """Load a saved model from disk.

        Args:
            path: Path to a ``.pkl`` file previously written by :meth:`save`.

        Returns:
            A ready-to-use :class:`TIMEDClassifierRF` instance.
        """
        with open(path, "rb") as f:
            model = pickle.load(f)
        # R ensemble_weights objects don't survive pickle reliably.
        # Reset them so generate_predictions recomputes from scratch on first call.
        model.ensemble_weights = [None] * len(model.ensemble_weights)
        return cls(model)
