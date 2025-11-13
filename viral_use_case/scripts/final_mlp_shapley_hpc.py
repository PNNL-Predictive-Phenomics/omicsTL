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

seed_idx = int(sys.argv[1])
shap_samp_size = int(sys.argv[2])

source_input = all_datasets.source_data.drop(columns = ['Resp'])
target_input = all_datasets.target_test_data[0].drop(columns = ['Resp'])

from omicstl.simulation_utils.model_utils import create_data_partition

# MLP shapley Computation begin
def mlp_shap_pred_transfer(X):

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

def mlp_shap_pred_notransfer(X):

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
    preds1 = mlp_model_targetonly.__getattribute__("target_nosource").predict([dpart.features], 
                                                          return_probabilities = True)
    
    return preds1

random.seed(seed_idx)
torch.manual_seed(seed_idx)
mlp_explainer_xfer = shap.KernelExplainer(mlp_shap_pred_transfer, source_input)
mlp_shap_values_xfer = mlp_explainer_xfer.shap_values(target_input, nsamples = shap_samp_size)

mlp_shap_values_xfer0 = mlp_shap_values_xfer[:, :, 0]
mlp_shap_values_xfer1 = mlp_shap_values_xfer[:, :, 1]

mlp_df_xfer0 = pd.DataFrame(mlp_shap_values_xfer0)
mlp_df_xfer1 = pd.DataFrame(mlp_shap_values_xfer1)

mlp_df_xfer0.columns = target_input.columns
mlp_df_xfer1.columns = target_input.columns

mlp_outfile_xfer0 = f"/qfs/people/flor829/PPI_TIMED/timed-hpc/viral_use_case/results/shapley/mlp_shap_xfer0_seed{seed_idx}_shapsize{shap_samp_size}.csv"
mlp_outfile_xfer1 = f"/qfs/people/flor829/PPI_TIMED/timed-hpc/viral_use_case/results/shapley/mlp_shap_xfer1_seed{seed_idx}_shapsize{shap_samp_size}.csv"

mlp_df_xfer0.to_csv(mlp_outfile_xfer0)
mlp_df_xfer1.to_csv(mlp_outfile_xfer1) 



random.seed(seed_idx)
torch.manual_seed(seed_idx)
mlp_explainer_noxfer = shap.KernelExplainer(mlp_shap_pred_notransfer, source_input)
mlp_shap_values_noxfer = mlp_explainer_noxfer.shap_values(target_input, nsamples = shap_samp_size)

mlp_shap_values_noxfer0 = mlp_shap_values_noxfer[:, :, 0]
mlp_shap_values_noxfer1 = mlp_shap_values_noxfer[:, :, 1]

mlp_df_noxfer0 = pd.DataFrame(mlp_shap_values_noxfer0)
mlp_df_noxfer1 = pd.DataFrame(mlp_shap_values_noxfer1)

mlp_df_noxfer0.columns = target_input.columns
mlp_df_noxfer1.columns = target_input.columns

mlp_outfile_noxfer0 = f"/qfs/people/flor829/PPI_TIMED/timed-hpc/viral_use_case/results/shapley/mlp_shap_noxfer0_seed{seed_idx}_shapsize{shap_samp_size}.csv"
mlp_outfile_noxfer1 = f"/qfs/people/flor829/PPI_TIMED/timed-hpc/viral_use_case/results/shapley/mlp_shap_noxfer1_seed{seed_idx}_shapsize{shap_samp_size}.csv"

mlp_df_noxfer0.to_csv(mlp_outfile_noxfer0)
mlp_df_noxfer1.to_csv(mlp_outfile_noxfer1) 