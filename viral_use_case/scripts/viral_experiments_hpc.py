import pandas as pd
import random
import itertools
import sys
from tqdm import trange

from sklearn.model_selection import train_test_split
import torch
from torch import device

from omicstl.simulation_utils.data_utils import DatasetContainer
from omicstl.simulation_utils.model_utils import fit_dl_model
from omicstl.transfer_forest import load_r_functions
from omicstl.simulation_utils.model_utils import fit_rf_model

# Read in source, target_transfer, and target_validation datasets.
base_source_data = pd.read_csv('/qfs/people/flor829/PPI_TIMED/timed-hpc/viral_use_case/data/source_dset.csv').set_index("SampleID")
base_source_data['Resp'] = base_source_data['Resp'].apply(lambda x: 2 if x == 'viral' else 1)

base_target_transfer_data = pd.read_csv('/qfs/people/flor829/PPI_TIMED/timed-hpc/viral_use_case/data/target_transfer.csv').set_index("SampleID")
base_target_transfer_data['Resp'] = base_target_transfer_data['Resp'].apply(lambda x: 2 if x == 'viral' else 1)

base_target_validation_data = pd.read_csv('/qfs/people/flor829/PPI_TIMED/timed-hpc/viral_use_case/data/target_validation.csv').set_index("SampleID")
base_target_validation_data['Resp'] = base_target_validation_data['Resp'].apply(lambda x: 2 if x == 'viral' else 1)

de_prots_info = pd.read_csv('/qfs/people/flor829/PPI_TIMED/timed-hpc/viral_use_case/data/de_prots.csv')

# Extract protein names that are differentially expressed
de_prots = de_prots_info.loc[de_prots_info['flag_target'].isin([1, -1]), 'Protein'].tolist()
nonde_prots = de_prots_info.loc[de_prots_info['flag_target'].isin([0]), 'Protein'].tolist()

total_prots = base_source_data.shape[1] - 1 # substract 1 to remove the response variable

# As a reminder, in our experiments...
# 1. We will consider various sizes of the base "source" dataset to use for initial training
# 2. We will consider various sizes of the transer partition for the target dataset to use for transer modeling
# 3. As a proxy for SNR, we will consider various ratios of differentially expressed to non-differentially expressed features in 
#    the source/target-transfer datasets.
# 4. Experiments that vary the above three conditions will be done on datasets compiled using multiple timepoints, but we will
#    perform similar experiments on datasets compiled using only one timepoint (in case the modeling is substantially affected
#    by our ignoring timepoint in modeling efforts)

# We define a pandas dataframe that represents all combinations of
# the parameters referenced above in (1-3). For (4), a separate
# script is used that pre-filters the dataset to a specific timepoint
# but otherwise implements the same experiment. 

# Define the inputs
source_sizes = [15, 25, 100, 358]
target_sizes = [15, 25, 100, 192]
snrs = [0/total_prots, 1/total_prots, 5/total_prots, 25/total_prots, 68/total_prots, 68/68]

exp_conditions = list(itertools.product(source_sizes, target_sizes, snrs))

exp_conditions_df = pd.DataFrame(exp_conditions, columns=["source_size", "target_size", "snr"])


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
idx = int(sys.argv[1])-1
nreps = int(sys.argv[2])

print(idx)
print(nreps)

# idx = 0 # the index of the set of experiment conditions
# nreps = 10 # the number of replicates for the given experiment scenario

# Extract the experimental parameter values at the indicated idx
source_size = exp_conditions_df.loc[idx, 'source_size']
target_size = exp_conditions_df.loc[idx, 'target_size']
snr = exp_conditions_df.loc[idx, 'snr']
num_de_prots = min(snr * total_prots, len(de_prots))
num_nonde_prots = 0 if snr == 1 else total_prots

# Begin Loop --------------------------------------------------------------
random.seed(idx)
for i in range(0, nreps):

    # Sample which de_prots are actually selected. 
    loop_de_prots = random.sample(de_prots, int(num_de_prots)) if snr != 1 else de_prots
    loop_nonde_prots = nonde_prots if snr != 1 else []

    # Sample which samples are actually selected
    if float(source_size/base_source_data.shape[0]) == 1:
        loop_source_data = base_source_data
    else:
        loop_source_data, _ = train_test_split(base_source_data, train_size=float(source_size/base_source_data.shape[0]),
                                               stratify=base_source_data['Resp'])
        
    if float(target_size/base_target_transfer_data.shape[0]) == 1:
        loop_transfer_data = base_target_transfer_data
    else:
        loop_transfer_data, _ = train_test_split(base_target_transfer_data, train_size=float(target_size/base_target_transfer_data.shape[0]), 
                                                 stratify=base_target_transfer_data['Resp'])

    # Down-select proteins in the above-defined datasets. 
    loop_vars = ['Resp'] + loop_de_prots + loop_nonde_prots

    loop_source_data = loop_source_data[loop_vars]
    loop_transfer_data = loop_transfer_data[loop_vars]
    loop_validation_data = base_target_validation_data[loop_vars]

    try:
        loop_datasets = DatasetContainer(
                source_data = loop_source_data,
                target_data = loop_transfer_data,
                target_test_data = [loop_validation_data]
                )
        loop_datasets.set_response_column("Resp") # Identify response column
    except:
        print(f"Error setting up dataset for condition set={idx}, replicate={i}")

    # Define where to save output
    mlp_outfile = f"/qfs/people/flor829/PPI_TIMED/timed-hpc/viral_use_case/results/mlp_expcond_{idx}_replicate_{i}.csv"
    vae_outfile = f"/qfs/people/flor829/PPI_TIMED/timed-hpc/viral_use_case/results/vae_expcond_{idx}_replicate_{i}.csv"
    rf_outfile = f"/qfs/people/flor829/PPI_TIMED/timed-hpc/viral_use_case/results/rf_expcond_{idx}_replicate_{i}.csv"

    # For modeling only, set a constant seed.
    torch.manual_seed(42)
    try:
        loop_mlp_out, loop_mlp_model, loop_mlp_model_targetonly = fit_dl_model(
            loop_datasets,
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
            loop_datasets,
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
        loop_rf_out, loop_rf_model = fit_rf_model(loop_datasets)
        loop_rf_out.to_csv(rf_outfile)
    except Exception as e:
        print(f"Failed RF for: `{rf_outfile}")
        print(e)

# End Loop --------------------------------------------------------------