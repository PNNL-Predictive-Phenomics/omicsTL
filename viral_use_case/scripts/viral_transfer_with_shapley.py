import pandas as pd
import random
import itertools
import sys
from tqdm import trange
import shap
import numpy as np

from sklearn.model_selection import train_test_split
import torch
from torch import device

from omicstl.simulation_utils.data_utils import DatasetContainer
from omicstl.simulation_utils.model_utils import fit_dl_model
from omicstl.transfer_forest import load_r_functions
from omicstl.simulation_utils.model_utils import fit_rf_model

# Read in source, target_transfer, and target_validation datasets.
# base_source_data = pd.read_csv('/qfs/people/flor829/PPI_TIMED/timed-hpc/viral_use_case/data/source_dset.csv').set_index("SampleID")
base_source_data = pd.read_csv('/workspaces/timed-hpc/viral_use_case/data/source_dset.csv').set_index("SampleID")
base_source_data['Resp'] = base_source_data['Resp'].apply(lambda x: 2 if x == 'viral' else 1)

# base_target_transfer_data = pd.read_csv('/qfs/people/flor829/PPI_TIMED/timed-hpc/viral_use_case/data/target_transfer.csv').set_index("SampleID")
base_target_transfer_data = pd.read_csv('/workspaces/timed-hpc/viral_use_case/data/target_transfer.csv').set_index("SampleID")
base_target_transfer_data['Resp'] = base_target_transfer_data['Resp'].apply(lambda x: 2 if x == 'viral' else 1)

# base_target_validation_data = pd.read_csv('/qfs/people/flor829/PPI_TIMED/timed-hpc/viral_use_case/data/target_validation.csv').set_index("SampleID")
base_target_validation_data = pd.read_csv('/workspaces/timed-hpc/viral_use_case/data/target_validation.csv').set_index("SampleID")
base_target_validation_data['Resp'] = base_target_validation_data['Resp'].apply(lambda x: 2 if x == 'viral' else 1)

# These are the parameters used previously for the deep learning models.
# Depending on Evan's findings, these may need to be changed. 
param_grid = {
        "dropout": [0.25, 0.5],
        "n_latent_dims": [2],
        "hidden_dim_base": [6],
        "lr": [0.01, 0.001],
        "source_epochs": [1000],
        "target_epochs": [1000],
        "freeze": ["none"],
        "weight_decay": [1e-4, 1e-2],
        "gamma": [1, 2, 3]
    }

random.seed(1123)

try:
    all_datasets = DatasetContainer(
            source_data = base_source_data,
            target_data = base_target_transfer_data,
            target_test_data = [base_target_validation_data]
            )
    all_datasets.set_response_column("Resp") # Identify response column
except:
    print(f"Error setting up dataset")

# For modeling only, set a constant seed.
torch.manual_seed(42)
try:
    mlp_out, mlp_model, mlp_model_targetonly = fit_dl_model(
        all_datasets,
        "mult_mlp",
        device("cpu"),
        param_grid
    )
except Exception as e:
    print(f"Failed MLP")
    print(e)

torch.manual_seed(42)
try:
    vae_out, vae_model, vae_model_targetonly = fit_dl_model(
        all_datasets,
        "mult_vae",
        device("cpu"),
        param_grid
    )
except Exception as e:
    print(f"Failed VAE")
    print(e)

random.seed(42)
try:
    rf_out, rf_model = fit_rf_model(all_datasets)
except Exception as e:
    print(f"Failed RF")
    print(e)

source_input = all_datasets.source_data.drop(columns = ['Resp'])
target_input = all_datasets.target_test_data[0].drop(columns = ['Resp'])

from omicstl.simulation_utils.model_utils import create_data_partition

# MLP shapley Computation begin
def mlp_shap_pred(X):

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

    X_copy['Resp'] = 1 # create a dummy outcome for the sake of create_data_partition
    full_ids = X_copy.columns.tolist()
    resp_id = 'Resp' 
    feats_ids = full_ids[:-1] # assumes the column name of the response variable is always last.

    # now create the data partition
    dpart = create_data_partition(
        data=X_copy,
        response_id=resp_id,
        feature_cols=feats_ids
        )

    # Now get full set of predictions. Here we are just going
    # to go for the target (i.e. tranfer) vae model.
    # You would need to hard-code differently should you want 
    # probabilities from the other models (i.e. target_nosource or source
    # instead of target). 
    preds1 = mlp_model.__getattribute__("target").predict([dpart.features], 
                                                          return_probabilities = True)
    
    return preds1

mlp_shap_pred(target_input)

# Now to do shapley stuff for random forest. 

random.seed(1)
torch.manual_seed(1)
mlp_explainer = shap.KernelExplainer(mlp_shap_pred, source_input)
mlp_shap_values = mlp_explainer.shap_values(target_input, nsamples = 1000)

mlp_shap_values0 = mlp_shap_values[:, :, 0]
mlp_shap_values1 = mlp_shap_values[:, :, 1]

mlp_df0 = pd.DataFrame(mlp_shap_values0)
mlp_df1 = pd.DataFrame(mlp_shap_values1)

mlp_df0.columns = target_input.columns
mlp_df1.columns = target_input.columns

random.seed(1)
torch.manual_seed(1)
mlp_explainer2 = shap.KernelExplainer(mlp_shap_pred, source_input.sample(10))
mlp_shap_values2 = mlp_explainer2.shap_values(target_input, nsamples = 10)

# MLP shapley Computation end



# VAE shapley Computation begin
def vae_shap_pred(X):

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

    X_copy['Resp'] = 1 # create a dummy outcome for the sake of create_data_partition
    full_ids = X_copy.columns.tolist()
    resp_id = 'Resp' 
    feats_ids = full_ids[:-1] # assumes the column name of the response variable is always last.

    # now create the data partition
    dpart = create_data_partition(
        data=X_copy,
        response_id=resp_id,
        feature_cols=feats_ids
        )
    
    # Now get full set of predictions. Here we are just going
    # to go for the target (i.e. tranfer) vae model.
    # You would need to hard-code differently should you want 
    # probabilities from the other models (i.e. target_nosource or source
    # instead of target). 
    preds1 = vae_model.__getattribute__("target").predict([dpart.features], 
                                                          return_probabilities = True)
    return preds1

source_input = all_datasets.source_data
target_input = all_datasets.target_test_data[0]

vae_shap_pred(target_input)

# Now to do shapley stuff for random forest. 

vae_explainer = shap.KernelExplainer(vae_shap_pred, source_input.sample(10))
vae_shap_values = vae_explainer.shap_values(target_input, nsamples = 10)

# VAE shapley Computation end






# Random Forest Shapley Computation begin

from omicstl.transfer_forest import TransferForest
import re
def rf_shap_pred(X):

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

    X_copy['Resp'] = 1 # create a dummy outcome for the sake of create_data_partition
    full_ids = X_copy.columns.tolist()
    resp_id = 'Resp' 
    feats_ids = full_ids[:-1] # assumes the column name of the response variable is always last.

    # now create the data partition
    dpart = create_data_partition(
        data=X_copy,
        response_id=resp_id,
        feature_cols=feats_ids
        )

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
    
    all_preds = all_preds[0]

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
    all_preds_reorg = {}

    # Iterate over each group, apply regex filters, and build dataframes
    for new_key, patterns in groupings.items():
        matched_keys = []
        for pattern in patterns:
            # Use regex to match keys in the dictionary
            matched_keys.extend([key for key in all_preds.keys() if re.match(pattern, key)])
                
        if matched_keys:
            # Create a pandas DataFrame by column-binding arrays corresponding to matched keys
            df = pd.DataFrame({key: all_preds[key] for key in matched_keys})
            all_preds_reorg[new_key] = df

    final_preds_df = all_preds_reorg.get("pred_ensemble_full")
    predicted_probs = final_preds_df.drop(columns = "pred_ensemble")
    
    return predicted_probs.to_numpy()

source_input = all_datasets.source_data
target_input = all_datasets.target_test_data[0]

rf_shap_pred(target_input)

# Now to do shapley stuff for random forest. 

rf_explainer = shap.KernelExplainer(rf_shap_pred, source_input.sample(10))
rf_shap_values = rf_explainer.shap_values(target_input, nsamples = 10)

# Random Forest Shapley Computation end


