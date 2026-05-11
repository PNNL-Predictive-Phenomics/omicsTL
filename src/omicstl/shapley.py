import numpy as np
from numpy.typing import NDArray
import pandas as pd
import shap
from omicstl.simulation_utils.model_utils import DataPartition, create_data_partition, reorg_rf_predictions
from omicstl.transfer_forest import TransferForest
from omicstl.transfer_networks import TransferMLP, TransferVAE

def _partition_input(X, source_input, resp_id) -> DataPartition:
    X_copy = X.copy()

    # SHAP does an automatic conversion here to a np array, even if
    # the provided input is a pandas dataframe.
    # Ensure the input is a pandas DataFrame (even if it's converted to numpy array by SHAP)
    if isinstance(X_copy, np.ndarray):
        # Convert X back to DataFrame
        # DUring the conversion have to hard-code column names 
        # based on what was specified externally for 'sample_input'.
        # Be sure to change 'sample_input' to whatever you are 
        # using as input for the data to shap.KernelExplainer(). 
        # I don't know of another solution around this problem.
        X_copy = pd.DataFrame(X_copy, columns=source_input.columns)

    X_copy = X_copy.drop(columns=[resp_id], errors='ignore') # Remove existing resp column if present
    X_copy[resp_id] = 1 # create a dummy outcome for the sake of create_data_partition
    full_ids = X_copy.columns.tolist()
    feats_ids = full_ids[:-1] # assumes the column name of the response variable is always last.

    # now create the data partition
    dpart = create_data_partition(
        data=X_copy,
        response_id=resp_id,
        feature_cols=feats_ids
        )
    
    return dpart

def _dl_shap_pred(X, source_input, mlp_model, resp_id):
    dpart = _partition_input(X, source_input, resp_id)

    # Now get full set of predictions. Here we are just going
    # to go for the target (i.e. tranfer) mlp model.
    # You would need to hard-code differently should you want 
    # probabilities from the other models (i.e. target_nosource or source
    # instead of target). 
    preds1 = mlp_model.__getattribute__("target").predict([dpart.features], 
                                                          return_probabilities = True)
    
    return preds1

def _rf_shap_pred(X, source_input, rf_model, resp_id):
    dpart = _partition_input(X, source_input, resp_id)

    # Now get full set of predictions. Here we are just going
    # to go for the ensemble version predicted probabilities.
    # You would need to hard-code differently should you want 
    # probabilities from the other models. 
    # Note that rf_model is hard-coded. It should be
    # changed to whatever you call the model object output
    # by fit_rf_model()
    all_preds = rf_model.generate_predictions(
        views=[dpart.features],
        response=dpart.response,
        validation_views=[dpart.features],
        validation_response=dpart.response,
        ensemble_views=None,
        ensemble_response=None,
        integration_type=TransferForest.IntegrationType.NONE
        )

    all_preds_reorg = reorg_rf_predictions(all_preds[0])
    final_preds_df = all_preds_reorg.get("pred_ensemble_full")
    
    # Classification
    if "pred_ensemble" in final_preds_df.columns and final_preds_df.shape[1] > 1:
        return final_preds_df.drop(columns="pred_ensemble").to_numpy()
    
    # Regression
    return final_preds_df[["pred_ensemble"]].to_numpy()

def _shap_to_df(
        shap_values: NDArray,
        target_input: pd.DataFrame
    ) -> pd.DataFrame | dict[int, pd.DataFrame]:
    """Handle different shapes that shap_values returns and convert them to a dictionary of
       dataframes identified by class for classification or a single dataframe for regression"""

    # Binary classification
    if isinstance(shap_values, list):
        return {
            class_id + 1: pd.DataFrame(arr, columns=target_input.columns, index=target_input.index)
            for class_id, arr in enumerate(shap_values)
        }
    
    # Multiclass
    if shap_values.ndim == 3: 
        return {
            class_id + 1: pd.DataFrame(shap_values[:, :, class_id], columns=target_input.columns, index=target_input.index)
            for class_id in range(shap_values.shape[2])
        }
    
    # Regression
    return pd.DataFrame(shap_values, columns=target_input.columns, index=target_input.index)


def get_shapley_values_dl(
        source_input: pd.DataFrame,
        target_input: pd.DataFrame,
        pretrained_model: TransferMLP | TransferVAE,
        background_sample_size: int | None = None,
        shap_sample_size: int | str = "auto",
        response_id: str = "Resp"
    ) -> pd.DataFrame | dict[int, pd.DataFrame]:
    """Get Shapley values for a Deep Learning model

    Args:
        source_input: a DataFrame containing the source data used to train the pretrained_model
        target_input: a DataFrame containing samples you want to explain from the same domain
            as the target data used to train the pretrained_model.
        pretrained_model: a transfer learning model created using fit_rf_model
        background_sample_size: (optional) Number of source samples to use for SHAP reference.
            Default is all samples, but this can get very slow.
        shap_sample_size: (optional) The number of feature arrangements used by SHAP to estimate
            the contribution of each feature. Default is "auto", which allows SHAP to select a
            reasonable amount given the data.
        response_id: (optional) The response ID column name. Defaults to "Resp"
    
    Returns:
        A dataframe of Shapley values for regression or a dictionary of dataframes indexed by
        the prediction class for classification.
    """
    sampled_input = source_input.sample(background_sample_size) if background_sample_size is not None else source_input
    explainer = shap.KernelExplainer(lambda X: _dl_shap_pred(X, source_input, pretrained_model, response_id), sampled_input)
    shap_values = explainer.shap_values(target_input, nsamples = shap_sample_size)
    return _shap_to_df(shap_values, target_input)

def get_shapley_values_rf(
        source_input: pd.DataFrame,
        target_input: pd.DataFrame,
        pretrained_model: TransferForest,
        background_sample_size: int | None = None,
        shap_sample_size: int | str = "auto",
        response_id: str = "Resp"
    ) -> pd.DataFrame | dict[int, pd.DataFrame]:
    """Get Shapley values for a Random Forest model

    Args:
        source_input: a DataFrame containing the source data used to train the pretrained_model
        target_input: a DataFrame containing samples you want to explain from the same domain
            as the target data used to train the pretrained_model.
        pretrained_model: a transfer learning model created using fit_rf_model
        background_sample_size: (optional) Number of source samples to use for SHAP reference.
            Default is all samples, but this can get very slow.
        shap_sample_size: (optional) The number of feature arrangements used by SHAP to estimate
            the contribution of each feature. Default is "auto", which allows SHAP to select a
            reasonable amount given the data.
        response_id: (optional) The response ID column name. Defaults to "Resp"
    
    Returns:
        A dataframe of Shapley values for regression or a dictionary of dataframes indexed by
        the prediction class for classification.
    """
    sampled_input = source_input.sample(background_sample_size) if background_sample_size is not None else source_input
    explainer = shap.KernelExplainer(lambda X: _rf_shap_pred(X, source_input, pretrained_model, response_id), sampled_input)
    shap_values = explainer.shap_values(target_input, nsamples = shap_sample_size)
    return _shap_to_df(shap_values, target_input)