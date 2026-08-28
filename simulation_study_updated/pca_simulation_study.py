import os
import sys
import random
import itertools

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import device

from omicstl.simulation_utils.data_utils import DatasetContainer
from omicstl.simulation_utils.model_utils import fit_dl_model, fit_rf_model
from omicstl.simulation_utils.data_generation import generate_synth_data_pca, generate_pca_response

# -----------------------
# Safety: required args
# -----------------------
if len(sys.argv) < 4:
    raise ValueError(
        f"Expected args: <task_id> <nreps> <setno>. Got: {sys.argv}"
    )

# -----------------------
# Paths
# -----------------------
USER = "obir854"
DATA_DIR = f"/qfs/people/{USER}/ppi_timed/data"
SOURCE_CSV = os.path.join(DATA_DIR, "source_dset.csv")
TARGET_CSV = os.path.join(DATA_DIR, "target_transfer.csv")
OUT_DIR = "/qfs/projects/ppi_timed/sim_results_v2/results"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------
# Read data (features only — response is generated synthetically below)
# -----------------------
base_source_data = pd.read_csv(SOURCE_CSV).set_index("SampleID").drop("Resp", axis=1)
base_target_data = pd.read_csv(TARGET_CSV).set_index("SampleID").drop("Resp", axis=1)

# -----------------------
# Experiment grid
#
# Removed vs previous script:
#   feature_ratios   — replaced by n_output_features=None in generate_synth_data_pca,
#                      which keeps all real-data features. PCA preserves the true
#                      covariance structure so an arbitrary ratio is no longer needed.
#   snrs_source/snrs_target — generate_synth_data_pca has no SNR parameter; noise is
#                      implicit in the PCA reconstruction from real data.
# -----------------------
source_sizes             = [100, 250, 500]
target_sizes             = [10, 25, 50, 100]
response_fn_options      = ["cont", "cat"]
response_fn_complexities = ["linear", "nonlinear"]
snr_values               = [1, 3, 5]   # ceilings: R^2 = 0.50, 0.75, 0.83
alpha_values             = [0.0, 0.5, 1.0]  # domain-shift interpolation

exp_conditions = list(itertools.product(
    source_sizes,
    target_sizes,
    response_fn_options,
    response_fn_complexities,
    snr_values,
    alpha_values,
))

exp_conditions_df = pd.DataFrame(
    exp_conditions,
    columns=[
        "source_size",
        "target_size",
        "response_fn_option",
        "response_fn_complexity",
        "snr",
        "alpha",
    ],
)

# Total conditions: 3 x 4 x 2 x 2 x 2 x 3 = 288
print(f"Total experiment conditions: {len(exp_conditions_df)}")

# -----------------------
# Deep learning param grid
# -----------------------
param_grid = {
    "dropout":         [0.25, 0.5],
    "n_latent_dims":   [4],
    "hidden_dim_base": [32, 64],
    "z_dim_base":      [12, 32],
    "lr":              [0.01, 0.001],
    "source_epochs":   [500],
    "target_epochs":   [500],
    "freeze":          ["none"],
    "weight_decay":    [1e-4, 1e-2],
    "gamma":           [1, 2, 3],
}

# Logistic separation strength for categorical response
# (SNR is ignored for categorical; gamma=3 gives clear but realistic class separation)
_CAT_GAMMA = 3

# -----------------------
# CLI args from SLURM
# -----------------------
idx  = int(sys.argv[1]) - 1       # convert 1-based task id to 0-based
nreps = int(sys.argv[2])
setno = int(sys.argv[3])
idx  = idx + 1000 * (setno - 1)  # offset for multi-set runs

print(f"idx={idx}  nreps={nreps}  setno={setno}")

source_size            = exp_conditions_df.loc[idx, "source_size"].item()
target_size            = exp_conditions_df.loc[idx, "target_size"].item()
response_fn_option     = exp_conditions_df.loc[idx, "response_fn_option"]
response_fn_complexity = exp_conditions_df.loc[idx, "response_fn_complexity"]
snr                    = exp_conditions_df.loc[idx, "snr"].item()
alpha                  = exp_conditions_df.loc[idx, "alpha"].item()

print(f"source_size={source_size}  target_size={target_size}  "
      f"response={response_fn_option}  complexity={response_fn_complexity}  "
      f"snr={snr}  alpha={alpha}")

# Limit intra-op threads so parallel SLURM array tasks don't over-subscribe node CPUs
torch.set_num_threads(1)

# -----------------------
# Main replication loop
# -----------------------
for i in range(nreps):

    # ---- Seed everything before data generation so each replicate is reproducible
    rep_seed = int(idx) * 1000 + i
    np.random.seed(rep_seed)
    random.seed(rep_seed)
    torch.manual_seed(rep_seed)

    # ---- Generate synthetic features via PCA (shared source loadings)
    try:
        synth_source_feats, synth_target_feats, src_scores, tgt_scores = generate_synth_data_pca(
            source_data=base_source_data,
            target_data=base_target_data,
            n_output_samps_source=source_size,
            n_output_samps_target=target_size + 100,
            n_output_features=None,
            alpha=alpha,
        )
    except Exception as e:
        print(f"PCA generation failed for idx={idx}, rep={i}: {e}")
        continue

    # ---- Generate response directly from sampled PC scores (no PCA refit)
    is_cat = (response_fn_option == "cat")
    source_synth_data, target_synth_data, _ = generate_pca_response(
        source_features=synth_source_feats,
        target_features=synth_target_feats,
        complexity=response_fn_complexity,
        is_categorical=is_cat,
        snr=snr,
        source_scores=src_scores,
        target_scores=tgt_scores,
        gamma=_CAT_GAMMA if is_cat else 1.0,
    )

    # ---- Standardize continuous response using source training set stats
    if not is_cat:
        resp_mean = source_synth_data["response"].mean()
        resp_sd   = source_synth_data["response"].std(ddof=1)
        if resp_sd > 1e-10:
            source_synth_data = source_synth_data.copy()
            target_synth_data = target_synth_data.copy()
            source_synth_data["response"] = (source_synth_data["response"] - resp_mean) / resp_sd
            target_synth_data["response"] = (target_synth_data["response"] - resp_mean) / resp_sd

    # ---- Train / test split on target (100 held-out test samples)
    split_kwargs: dict = dict(train_size=target_size, test_size=100)
    if is_cat:
        split_kwargs["stratify"] = target_synth_data["response"]
    try:
        target_synth_train, target_synth_eval = train_test_split(
            target_synth_data, **split_kwargs
        )
    except ValueError as e:
        print(f"Train/test split failed for idx={idx}, rep={i}: {e}")
        continue

    # ---- Guard: single-class target makes classification undefined
    if is_cat and target_synth_train["response"].nunique() < 2:
        print(f"  Skipping idx={idx} rep={i}: only one class in target train (target_size={target_size})")
        continue

    # ---- Wrap into DatasetContainer
    try:
        datasets_sim = DatasetContainer(
            source_data=source_synth_data,
            target_data=target_synth_train,
            target_ensemble_data=None,
            target_test_data=[target_synth_eval],
        )
        datasets_sim.set_response_column("response")
    except Exception as e:
        print(f"DatasetContainer failed for idx={idx}, rep={i}: {e}")
        continue

    # ---- Output paths
    mlp_outfile = os.path.join(OUT_DIR, f"mlp_expcond_{idx}_replicate_{i}.csv")
    vae_outfile = os.path.join(OUT_DIR, f"vae_expcond_{idx}_replicate_{i}.csv")
    rf_outfile  = os.path.join(OUT_DIR, f"rf_expcond_{idx}_replicate_{i}.csv")

    # ---- Condition metadata added to every output for self-contained CSVs
    def annotate(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["exp_idx"]                = idx
        df["replicate"]              = i
        df["rep_seed"]               = rep_seed
        df["source_size"]            = source_size
        df["target_size"]            = target_size
        df["response_fn_option"]     = response_fn_option
        df["response_fn_complexity"] = response_fn_complexity
        df["snr"]                    = snr
        df["alpha"]                  = alpha
        return df

    # ---- Fit MLP
    torch.manual_seed(rep_seed)
    try:
        mlp_out, _, _ = fit_dl_model(datasets_sim, "mult_mlp", device("cpu"), param_grid)
        annotate(mlp_out).to_csv(mlp_outfile, index=False)
    except Exception as e:
        print(f"MLP failed for {mlp_outfile}: {e}")

    # ---- Fit VAE
    torch.manual_seed(rep_seed)
    try:
        vae_out, _, _ = fit_dl_model(datasets_sim, "mult_vae", device("cpu"), param_grid)
        annotate(vae_out).to_csv(vae_outfile, index=False)
    except Exception as e:
        print(f"VAE failed for {vae_outfile}: {e}")

    # ---- Fit RF
    random.seed(rep_seed)
    try:
        rf_out, _ = fit_rf_model(datasets_sim)
        annotate(rf_out).to_csv(rf_outfile, index=False)
    except Exception as e:
        print(f"RF failed for {rf_outfile}: {e}")