import pandas as pd
import numpy as np
import random
import itertools
import sys
from tqdm import trange
import math

from sklearn.model_selection import train_test_split
import torch
from torch import device

from omicstl.simulation_utils.data_utils import DatasetContainer
from omicstl.simulation_utils.model_utils import fit_dl_model
from omicstl.transfer_forest import load_r_functions
from omicstl.simulation_utils.model_utils import fit_rf_model
from omicstl.simulation_utils.data_generation import response_function, generate_synth_data

# Read in source, target_transfer, and target_validation datasets.
# base_source_data = pd.read_csv('/qfs/people/flor829/PPI_TIMED/timed-hpc/viral_use_case/data/source_dset.csv').set_index("SampleID")
base_source_data = pd.read_csv('/workspaces/timed-hpc/viral_use_case/data/source_dset.csv').set_index("SampleID")
base_source_data = base_source_data.drop('Resp', axis=1)

# base_target_transfer_data = pd.read_csv('/qfs/people/flor829/PPI_TIMED/timed-hpc/viral_use_case/data/target_transfer.csv').set_index("SampleID")
base_target_transfer_data = pd.read_csv('/workspaces/timed-hpc/viral_use_case/data/target_transfer.csv').set_index("SampleID")
base_target_transfer_data = base_target_transfer_data.drop('Resp', axis=1)

# We define a pandas dataframe that represents all combinations of
# the following parameters:
# size of source dataset: 25, 50, 100, 200
# size of target dataset: 5, 10, 25
# response function: continuous or categorical (binary)
# response complexity: nonlinear vs linear function of predictors
# SNR source: 0.1, 0.5, 1, 2
# SNR target: 0.1, 0.5, 1, 2
# number of features: (0.5, 1, 2) * max(source sample size, target sample size) 

# Define the inputs
source_sizes = [25, 50, 100, 200]
target_sizes = [5, 10, 25, 192]
response_fn_options = ["cont", "cat"]
response_fn_complexities = ["linear", "nonlinear"]
snrs_source = [0.1, 0.5, 1, 2]
snrs_target = [0.1, 0.5, 1, 2]
feature_ratios = [0.5, 1, 2]

exp_conditions = list(itertools.product(source_sizes, target_sizes, response_fn_options,
                                        response_fn_complexities, snrs_source, snrs_target,
                                        feature_ratios))

exp_conditions_df = pd.DataFrame(exp_conditions, columns=["source_size", "target_size", 
                                                          "response_fn_options", 
                                                          "response_fn_complexity",
                                                          "snr_source", "snr_target",
                                                          "feature_ratio"])

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


# The following will go either into a for-loop (if doing locally) or 
# as-is, but accepting command line argument values as per an HPC 
# implementation for processing in parallel. 

# Uncomment if running in the context of the HPC.
# sys.argv[0] is the script name, hence why we start at 1
# idx = int(sys.argv[1])-1
# nreps = int(sys.argv[2])

idx = 3067
nreps = 2

print(idx)
print(nreps)

# Extract the experimental parameter values at the indicated idx
source_size = exp_conditions_df.loc[idx, 'source_size'].item()
target_size = exp_conditions_df.loc[idx, 'target_size'].item()
response_fn_option = exp_conditions_df.loc[idx, 'response_fn_options']
response_fn_complexity = exp_conditions_df.loc[idx, 'response_fn_complexity']
snr_source = exp_conditions_df.loc[idx, 'snr_source'].item()
snr_target = exp_conditions_df.loc[idx, 'snr_target'].item()
feature_ratio = exp_conditions_df.loc[idx, 'feature_ratio'].item()

# Define function according to choices above:
if response_fn_complexity == "nonlinear":
    response_fn = response_function("tanh(df[, 2]) + df[, 1] * df[, ncol(df)] ^ 2")
if response_fn_complexity == "linear":
    response_fn = response_function("df[, 2] + df[, 1] * df[, ncol(df)]")

num_features = math.floor(feature_ratio * max(source_size, target_size))

i = 0
# Begin Loop --------------------------------------------------------------
random.seed(idx)
for i in range(0, nreps):


    # Set up the data
    if response_fn_option == "cont":
        source_synth_data, source_lc_info, _ = generate_synth_data(
            data = base_source_data, # input data
            num_features = num_features, # number of output features
            num_samples = source_size, # number of output samples
            response_fn = response_fn, # response function
            snr = snr_source # signal to noise ratio
            )
        
        target_synth_data, _, _ = generate_synth_data(
            data = base_target_transfer_data, # input data
            num_features = num_features, # number of output features
            num_samples = target_size + 100, # we add 100 here to use for held out evaluation
            response_fn = response_fn, # response function
            snr = snr_target, # signal to noise ratio
            prior_lc_info = source_lc_info # crucial to include else the source and target datasets aren't linked!
            )
        source_synth_data.rename(columns={source_synth_data.columns[0]: 'response'}, inplace=True)
        target_synth_data.rename(columns={target_synth_data.columns[0]: 'response'}, inplace=True)

        target_synth_train, target_synth_eval = train_test_split(
        target_synth_data,
        train_size = target_size,
        test_size = 100
        )
        
    if response_fn_option == "cat":
        source_synth_data, source_lc_info, _ = generate_synth_data(
            data = base_source_data, # input data
            num_features = num_features, # number of output features
            num_samples = source_size, # number of output samples
            response_fn = response_fn, # response function
            snr = snr_source, # signal to noise ratio
            response_parameters={
                "ncats": 2,
                "quantile": "quantile"
                }
            )
        
        target_synth_data, _, _ = generate_synth_data(
            data = base_target_transfer_data, # input data
            num_features = num_features, # number of output features
            num_samples = target_size + 100, # number of output samples
            response_fn = response_fn, # response function
            snr = snr_target, # signal to noise ratio
            prior_lc_info = source_lc_info, # crucial to include else the source and target datasets aren't linked!
            response_parameters = {
                "ncats": 2,
                "quantile": "quantile"
                }
            )
        
        source_synth_data.rename(columns={source_synth_data.columns[0]: 'response'}, inplace=True)
        target_synth_data.rename(columns={target_synth_data.columns[0]: 'response'}, inplace=True)
        source_synth_data = source_synth_data.astype({"response": np.int64})
        target_synth_data = target_synth_data.astype({"response": np.int64})

        target_synth_train, target_synth_eval = train_test_split(
        target_synth_data,
        train_size = target_size,
        test_size = 100,
        stratify = target_synth_data['response']
        )
    
    try:
        datasets_sim = DatasetContainer(
            source_data = source_synth_data,
            target_data = target_synth_train,
            target_ensemble_data=None, # If we were to specify data here, it would need to be a further partition of target_synth_train because otherwise implies a larger set of target data for training the model.
            target_test_data=[target_synth_eval]
            )
        datasets_sim.set_response_column("response")
    except:
        print(f"Error setting up dataset for condition set={idx}, replicate={i}")
    
    # Define where to save output
    # mlp_outfile = f"/qfs/people/flor829/PPI_TIMED/timed-hpc/viral_use_case/results/mlp_expcond_{idx}_replicate_{i}.csv"
    # vae_outfile = f"/qfs/people/flor829/PPI_TIMED/timed-hpc/viral_use_case/results/vae_expcond_{idx}_replicate_{i}.csv"
    # rf_outfile = f"/qfs/people/flor829/PPI_TIMED/timed-hpc/viral_use_case/results/rf_expcond_{idx}_replicate_{i}.csv"
    mlp_outfile = f"/workspaces/timed-hpc/simulation_study_updated/results/mlp_expcond_{idx}_replicate_{i}.csv"
    vae_outfile = f"/workspaces/timed-hpc/simulation_study_updated/results/vae_expcond_{idx}_replicate_{i}.csv"
    rf_outfile = f"/workspaces/timed-hpc/simulation_study_updated/results/rf_expcond_{idx}_replicate_{i}.csv"

    # For modeling only, set a constant seed.
    torch.manual_seed(42)
    try:
        loop_mlp_out, loop_mlp_model, loop_mlp_model_targetonly = fit_dl_model(
            datasets_sim,
            "mult_mlp",
            device("cpu"),
            param_grid
        )
        loop_mlp_out.to_csv(mlp_outfile) 
    except Exception as e:
        print(f"Failed MLP for: `{mlp_outfile}`")
        print(e)

    torch.manual_seed(42)
    try:
        loop_vae_out, loop_vae_model, loop_vae_model_targetonly = fit_dl_model(
            datasets_sim,
            "mult_vae",
            device("cpu"),
            param_grid
        )
        loop_vae_out.to_csv(vae_outfile) 
    except Exception as e:
        print(f"Failed VAE for: `{vae_outfile}`")
        print(e)

    random.seed(42)
    try:
        loop_rf_out, loop_rf_model = fit_rf_model(datasets_sim)
        loop_rf_out.to_csv(rf_outfile)
    except Exception as e:
        print(f"Failed RF for: `{rf_outfile}")
        print(e)

# End Loop --------------------------------------------------------------