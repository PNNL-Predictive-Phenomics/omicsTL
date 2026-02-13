from omicstl.simulation_utils.model_utils import calculate_metrics
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, matthews_corrcoef, precision_score, recall_score, f1_score

def test_regression():
    truth = pd.Series([1.1, 2.2, 3.5, 5.3, 2.4, 1.1, 1.6, 1.8, 2.9, 2.8])
    pred = pd.Series([0.9, 2.4, 2.8, 3.2, 2.0, 1.1, 1.5, 1.8, 2.6, 2.7])

    metrics = calculate_metrics(truth, pred, False)

    assert metrics is not None

    assert metrics["rmse"] is not None
    assert abs(metrics["rmse"] - float(np.sqrt(np.mean((truth - pred) ** 2)))) < 0.0001

    assert metrics["mae"] is not None
    assert abs(metrics["mae"] - float(np.mean(np.abs(pred - truth)))) < 0.0001

    assert np.isnan(metrics["acc"])
    assert np.isnan(metrics["roc_auc"])
    assert np.isnan(metrics["mcc"])
    assert np.isnan(metrics["precision"])
    assert np.isnan(metrics["recall"])
    assert np.isnan(metrics["f1"])

def test_classification_multiclass():
    truth = pd.Series([1, 2, 3, 3, 2, 1, 2, 2, 3, 3])
    pred = pd.Series([1, 2, 2, 3, 2, 1, 1, 2, 2, 2])
    pred_probs = np.array([
        [0.9, 0.1, 0.1, 0.2, 0.2, 0.7, 0.3, 0.0, 0.0, 0.0],
        [0.1, 0.9, 0.5, 0.3, 0.6, 0.2, 0.7, 0.9, 0.55, 0.6],
        [0.0, 0.0, 0.4, 0.5, 0.2, 0.1, 0.0, 0.1, 0.45, 0.4]
    ]).transpose()

    metrics = calculate_metrics(truth, pred, True, pred_probs)

    assert metrics is not None

    assert np.isnan(metrics["rmse"])
    assert np.isnan(metrics["mae"])
    assert abs(metrics["acc"] - 0.6) < 0.0001

    roc_auc = roc_auc_score(truth, pred_probs, multi_class="ovr", average="weighted")
    assert abs(metrics["roc_auc"] - roc_auc) < 0.0001

    mcc = matthews_corrcoef(truth, pred)
    assert abs(metrics["mcc"] - mcc) < 0.0001

    precision = precision_score(truth, pred, average="weighted", zero_division=0)
    assert abs(metrics["precision"] - precision) < 0.0001
    
    recall = recall_score(truth, pred, average="weighted", zero_division=0)
    assert abs(metrics["recall"] - recall) < 0.0001

    f1 = f1_score(truth, pred, average="weighted", zero_division=0)
    assert abs(metrics["f1"] - f1) < 0.0001

def test_classification_binary():
    truth = pd.Series([1, 2, 2, 2, 2, 1, 2, 2, 2, 2])
    pred = pd.Series([1, 2, 2, 1, 2, 1, 1, 2, 2, 2])
    pred_probs = np.array([0.1, 0.9, 0.51, 0.7, 0.6, 0.3, 0.7, 0.9, 0.55, 0.6])

    metrics = calculate_metrics(truth, pred, True, pred_probs)

    assert metrics is not None

    assert np.isnan(metrics["rmse"])
    assert np.isnan(metrics["mae"])
    assert abs(metrics["acc"] - 0.8) < 0.0001

    roc_auc = roc_auc_score(truth, pred_probs)
    assert abs(metrics["roc_auc"] - roc_auc) < 0.0001

    tp = float(np.sum((truth == 2) & (pred == 2)))
    tn = float(np.sum((truth == 1) & (pred == 1)))
    fp = float(np.sum((truth == 1) & (pred == 2)))
    fn = float(np.sum((truth == 2) & (pred == 1)))

    print(tp, tn, fp, fn)
    print(metrics["precision"])
    print(metrics["recall"])
    print(metrics["mcc"])
    print(metrics["f1"])

    mcc = ((tp * tn) - (fp * fn)) / float(np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    assert abs(metrics["mcc"] - mcc) < 0.0001

    precision = (tp / (tp + fp))
    assert abs(metrics["precision"] - precision) < 0.0001

    recall = (tp / (tp + fn))
    assert abs(metrics["recall"] - recall) < 0.0001
    
    f1 = 2 * precision * recall / (precision + recall)
    assert abs(metrics["f1"] - f1) < 0.0001
