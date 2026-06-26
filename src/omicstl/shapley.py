import numpy as np
from numpy.typing import NDArray
import pandas as pd
import shap
import matplotlib.pyplot as plt
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
    ) -> dict[int, pd.DataFrame]:
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
    return {
        1: pd.DataFrame(shap_values, columns=target_input.columns, index=target_input.index)
    }


def get_shapley_values_dl(
        source_input: pd.DataFrame,
        target_input: pd.DataFrame,
        pretrained_model: TransferMLP | TransferVAE,
        background_sample_size: int | None = None,
        shap_sample_size: int | str = "auto",
        response_id: str = "Resp"
    ) -> dict[int | str, pd.DataFrame | dict]:
    """Get Shapley values for a Deep Learning model

    Args:
        source_input: a DataFrame containing the source data used to train the pretrained_model
        target_input: a DataFrame containing samples you want to explain from the same domain
            as the target data used to train the pretrained_model.
        pretrained_model: a transfer learning model created using fit_dl_model
        background_sample_size: (optional) Number of source samples to use for SHAP reference.
            Default is all samples, but this can get very slow.
        shap_sample_size: (optional) The number of feature arrangements used by SHAP to estimate
            the contribution of each feature. Default is "auto", which allows SHAP to select a
            reasonable amount given the data.
        response_id: (optional) The response ID column name. Defaults to "Resp"
    
    Returns:
        A dictionary of dataframes indexed by the prediction class. For regression, data is
        assigned to slot 1. A stored config is also attached.
    """

    # Drop response column
    source_clean = source_input.drop(columns=[response_id], errors='ignore')
    target_clean = target_input.drop(columns=[response_id], errors='ignore')

    # Calculate Shapley values
    sampled_input = source_clean.sample(background_sample_size) if background_sample_size is not None else source_clean
    explainer = shap.KernelExplainer(lambda X: _dl_shap_pred(X, source_clean, pretrained_model, response_id), sampled_input)
    shap_values = explainer.shap_values(target_clean, nsamples = shap_sample_size)
    result: dict[int | str, pd.DataFrame | dict] = _shap_to_df(shap_values, target_clean)

    # Add config for plotting and results interpretation
    result["config"] = dict(
        source_input=source_input,
        target_input=target_input,
        pretrained_model=pretrained_model,
        background_sample_size=background_sample_size,
        shap_sample_size=shap_sample_size,
        response_id=response_id
    )

    return result

def get_shapley_values_rf(
        source_input: pd.DataFrame,
        target_input: pd.DataFrame,
        pretrained_model: TransferForest,
        background_sample_size: int | None = None,
        shap_sample_size: int | str = "auto",
        response_id: str = "Resp"
    ) -> dict[int | str, pd.DataFrame | dict]:
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
        A dictionary of dataframes indexed by the prediction class. For regression, data is
        assigned to slot 1. A stored config is also attached.
    """

    # Drop response column
    source_clean = source_input.drop(columns=[response_id], errors='ignore')
    target_clean = target_input.drop(columns=[response_id], errors='ignore')

    # Calculate Shapley values
    sampled_input = source_clean.sample(background_sample_size) if background_sample_size is not None else source_clean
    explainer = shap.KernelExplainer(lambda X: _rf_shap_pred(X, source_clean, pretrained_model, response_id), sampled_input)
    shap_values = explainer.shap_values(target_clean, nsamples = shap_sample_size)
    result: dict[int | str, pd.DataFrame | dict] = _shap_to_df(shap_values, target_clean)

    # Add config for plotting and results interpretation
    result["config"] = dict(
        source_input=source_input,
        target_input=target_input,
        pretrained_model=pretrained_model,
        background_sample_size=background_sample_size,
        shap_sample_size=shap_sample_size,
        response_id=response_id
    )

    return result

def plot_shapley_values(
        shap_results: dict[int | str, pd.DataFrame | dict],
        class_map: dict[int, str] = {1: ""},
        max_display: int = 15,
        **kwargs
    ):
    """Plot SHAP summary.
    
    Args:
        shap_results: the results from get_shapley_values_rf or get_shapley_values_dl
        class_map: a dict mapping classes to class names. Only classes with keys in this dict will
            be plotted.
        max_display: the number of features to display
    """

    # Fast path for regression or binary classification
    if list(class_map.keys()) == [1]:
        feat_list = shap_results[1].columns.tolist()        

        shap.summary_plot(
            shap_results[1].values,
            shap_results["config"]["target_input"][feat_list],
            feature_names=feat_list,
            max_display=max_display,
            **kwargs
        )

        return
    
    # Calculate global feature importance (mean absolute SHAP across all classes)
    all_shap_values = pd.concat([
        shap_results[class_id].abs().mean() 
        for class_id in class_map.keys()
    ], axis=1).mean(axis=1)
    
    n_classes = len(class_map)
    top_features = all_shap_values.nlargest(max_display).index.tolist()
    
    # Create subplots
    fig, axes = plt.subplots(1, n_classes, figsize=(7 * n_classes, 8), sharey=True)
    if n_classes == 1:
        axes = [axes]
    
    for idx, (class_id, class_name) in enumerate(class_map.items()):
        # Filter to top features only
        shap_subset = shap_results[class_id][top_features]
        target_subset = shap_results["config"]["target_input"][top_features]
        
        plt.sca(axes[idx])
        shap.summary_plot(
            shap_subset.values,
            target_subset,
            feature_names=top_features,
            show=False,
            color_bar=(idx == n_classes - 1),  # Only show colorbar on rightmost plot
            **kwargs
        )
        
        axes[idx].set_title(class_name or f"Class {class_id}", fontsize=14, pad=10)
        axes[idx].set_xlabel("")
        if idx > 0:
            axes[idx].set_ylabel("")
        
        # Remove "Feature value" label from colorbar on rightmost plot
        if idx == n_classes - 1:
            cbar = plt.gcf().axes[-1]  # Get the colorbar axis
            cbar.set_ylabel("")  # Remove the label
    
    # Shared labels
    fig.text(0.5, 0.01, 'SHAP value (impact on model output)', ha='center', fontsize=12)
    fig.text(0.02, 0.5, 'Feature', va='center', rotation='vertical', fontsize=12)
    
    plt.tight_layout()
    plt.show()