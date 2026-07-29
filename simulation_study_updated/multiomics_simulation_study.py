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
from omicstl.simulation_utils.data_generation import (
    generate_synth_data_multiomics_latent,
    generate_pca_response,
)

# -----------------------
# Safety: required args
# -----------------------
if len(sys.argv) < 4:
    raise ValueError(
        f"Expected args: <task_id> <nreps> <setno>. Got: {sys.argv}"
    )

# -----------------------
# Paths  (override via env vars for local testing)
# -----------------------
MULTIOMICS_DIR = os.environ.get(
    "MULTIOMICS_DATA_DIR",
    "/qfs/people/obir854/ppi_timed/data/multiomics",
)
OUT_DIR = os.environ.get(
    "MULTIOMICS_OUT_DIR",
    "/qfs/projects/ppi_timed/multiomics_sim_results/results",
)
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------
# Load real data (features only — response generated synthetically)
# -----------------------
def load_omic(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0).drop(columns="response", errors="ignore")

base_source = {
    "prot":     load_omic(os.path.join(MULTIOMICS_DIR, "source_prot.csv")),
    "metab":    load_omic(os.path.join(MULTIOMICS_DIR, "source_metab.csv")),
    "lipidpos": load_omic(os.path.join(MULTIOMICS_DIR, "source_lipidpos.csv")),
    "lipidneg": load_omic(os.path.join(MULTIOMICS_DIR, "source_lipidneg.csv")),
}
base_target = {
    "prot":     load_omic(os.path.join(MULTIOMICS_DIR, "target_prot.csv")),
    "metab":    load_omic(os.path.join(MULTIOMICS_DIR, "target_metab.csv")),
    "lipidpos": load_omic(os.path.join(MULTIOMICS_DIR, "target_lipidpos.csv")),
    "lipidneg": load_omic(os.path.join(MULTIOMICS_DIR, "target_lipidneg.csv")),
}

# Combine lipidpos + lipidneg into a single "lipid" omic with prefixed column names
# so they can be treated as one view in single-omic conditions.
def _combine_lipids(omics: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pos = omics["lipidpos"].add_prefix("pos__")
    neg = omics["lipidneg"].add_prefix("neg__")
    return pd.concat([pos, neg], axis=1)

base_source["lipid"] = _combine_lipids(base_source)
base_target["lipid"] = _combine_lipids(base_target)

# -----------------------
# Experiment grid
#
# Factors:
#   source_sizes             3  : 100, 250, 500
#   target_sizes             4  : 10, 25, 50, 100
#   response_fn_options      2  : cont, cat
#   response_fn_complexities 2  : linear, nonlinear
#   snr_values               2  : 1, 5
#   shared_var_ratio_values  3  : 0.3, 0.6, 0.9
#   modality_values          4  : prot, metab, lipid, multi
#
# Total: 2 x 4 x 2 x 2 x 2 x 2 x 3 = 384 conditions
# -----------------------
source_sizes             = [250, 500]
target_sizes             = [10, 25, 50, 100]
response_fn_options      = ["cont", "cat"]
response_fn_complexities = ["linear", "nonlinear"]
snr_values               = [1, 5]
shared_var_ratio_values  = [0.3, 0.9]
modality_values          = ["prot", "metab", "multi"]

# Fixed: alpha=0 (no domain shift — focus on cross-disease TL only)
ALPHA = 0.0
# Fixed: shared latent dimensions driving all omic layers and the response
N_SHARED_FACTORS = 10
# Cap per-omic features before joint PCA so proteomics (~1000 proteins) does
# not drown out metabolomics/lipidomics (143–150+ features each).
# 150 leaves the smallest omic untouched (143 < 150), trims the 150+ omics by
# a handful, and cuts proteins from ~1000 to 150 — all layers end up ≈1:1.
TOP_N_PER_OMIC = 150

exp_conditions = list(itertools.product(
    source_sizes,
    target_sizes,
    response_fn_options,
    response_fn_complexities,
    snr_values,
    shared_var_ratio_values,
    modality_values,
))

exp_conditions_df = pd.DataFrame(
    exp_conditions,
    columns=[
        "source_size",
        "target_size",
        "response_fn_option",
        "response_fn_complexity",
        "snr",
        "shared_var_ratio",
        "modality",
    ],
)

assert len(exp_conditions_df) == 384, f"Expected 384 conditions, got {len(exp_conditions_df)}"
print(f"Total experiment conditions: {len(exp_conditions_df)}")

# -----------------------
# Deep learning param grid
# -----------------------
param_grid = {
    "dropout":         [0.25, 0.5],
    "n_latent_dims":   [4],
    "hidden_dim_base": [32, 64],   # 32→64 widens the fusion layer (MLP: concat 4×last_hidden → hidden_dim)
    "z_dim_base":      [12, 32],   # VAE-only: per-view variational bottleneck before PoE
    "lr":              [0.01, 0.001],
    "source_epochs":   [500],
    "target_epochs":   [500],
    "freeze":          ["none"],
    "weight_decay":    [1e-4, 1e-2],
    "gamma":           [1, 2, 3],
}

_CAT_GAMMA = 3

# -----------------------
# CLI args from SLURM
# -----------------------
idx   = int(sys.argv[1]) - 1
nreps = int(sys.argv[2])
setno = int(sys.argv[3])
idx   = idx + 1000 * (setno - 1)

print(f"idx={idx}  nreps={nreps}  setno={setno}")

source_size            = exp_conditions_df.loc[idx, "source_size"].item()
target_size            = exp_conditions_df.loc[idx, "target_size"].item()
response_fn_option     = exp_conditions_df.loc[idx, "response_fn_option"]
response_fn_complexity = exp_conditions_df.loc[idx, "response_fn_complexity"]
snr                    = exp_conditions_df.loc[idx, "snr"].item()
shared_var_ratio       = exp_conditions_df.loc[idx, "shared_var_ratio"].item()
modality               = exp_conditions_df.loc[idx, "modality"]

print(f"source_size={source_size}  target_size={target_size}  "
      f"response={response_fn_option}  complexity={response_fn_complexity}  "
      f"snr={snr}  shared_var_ratio={shared_var_ratio}  modality={modality}")

torch.set_num_threads(1)

# -----------------------
# Omic selection per modality
# -----------------------
# single-omic conditions: one-entry dict → generator fits PCA on that omic only
# multi: four-entry dict → generator fits joint PCA across all layers
MODALITY_KEYS = {
    "prot":  ["prot"],
    "metab": ["metab"],
    "lipid": ["lipid"],
    "multi": ["prot", "metab", "lipidpos", "lipidneg"],
}

selected_keys = MODALITY_KEYS[modality]
source_omics  = {k: base_source[k] for k in selected_keys}
target_omics  = {k: base_target[k] for k in selected_keys}


def build_flat_df(
    per_omic: dict[str, pd.DataFrame],
    response: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, list[str]] | None]:
    """Concatenate per-omic DataFrames into a flat sample x feature matrix.

    For multi-omics conditions, columns are prefixed with the omic name and
    view_column_groups is returned so DL models can process each omic via its
    own encoder. For single-omic conditions, no prefix is added and
    view_column_groups is None (single-view behaviour).
    """
    omic_list = list(per_omic.keys())
    is_multi = len(omic_list) > 1

    if is_multi:
        parts = []
        view_column_groups: dict[str, list[str]] = {}
        for k, df in per_omic.items():
            prefixed = df.add_prefix(f"{k}__")
            view_column_groups[k] = prefixed.columns.tolist()
            parts.append(prefixed)
        flat = pd.concat(parts, axis=1)
        flat.insert(0, "response", response)
        return flat, view_column_groups
    else:
        df = list(per_omic.values())[0].copy()
        df.insert(0, "response", response)
        return df, None


# -----------------------
# Main replication loop
# -----------------------
for i in range(nreps):

    rep_seed = int(idx) * 1000 + i
    np.random.seed(rep_seed)
    random.seed(rep_seed)
    torch.manual_seed(rep_seed)

    # ---- Generate synthetic multi-omics features via shared latent factors
    try:
        synth_src_omics, synth_tgt_omics, Z_src_syn, Z_tgt_syn = (
            generate_synth_data_multiomics_latent(
                source_omics=source_omics,
                target_omics=target_omics,
                n_output_samps_source=source_size,
                n_output_samps_target=target_size + 100,
                n_shared_factors=N_SHARED_FACTORS,
                shared_var_ratio=shared_var_ratio,
                alpha=ALPHA,
                snr=snr,
                random_state=rep_seed,
                top_n_per_omic=TOP_N_PER_OMIC,
            )
        )
    except Exception as e:
        print(f"Latent factor generation failed for idx={idx}, rep={i}: {e}")
        continue

    # ---- Generate response from the shared PC scores
    is_cat = (response_fn_option == "cat")

    # Flatten per-omic dicts into single feature DataFrames for response generation
    # (response is driven by the shared Z scores, so the feature content doesn't matter here)
    n_src_out = list(synth_src_omics.values())[0].shape[0]
    n_tgt_out = list(synth_tgt_omics.values())[0].shape[0]
    dummy_src_feat = pd.DataFrame(np.zeros((n_src_out, 1)), columns=["_dummy"])
    dummy_tgt_feat = pd.DataFrame(np.zeros((n_tgt_out, 1)), columns=["_dummy"])

    try:
        src_with_resp, tgt_with_resp, _ = generate_pca_response(
            source_features=dummy_src_feat,
            target_features=dummy_tgt_feat,
            complexity=response_fn_complexity,
            is_categorical=is_cat,
            snr=None,                    # SNR is already baked into features above
            source_scores=Z_src_syn,
            target_scores=Z_tgt_syn,
            gamma=_CAT_GAMMA if is_cat else 1.0,
        )
    except Exception as e:
        print(f"Response generation failed for idx={idx}, rep={i}: {e}")
        continue

    src_response = src_with_resp["response"].values
    tgt_response = tgt_with_resp["response"].values

    # ---- Build flat DataFrames with prefixed omic columns
    source_flat, view_groups = build_flat_df(synth_src_omics, src_response)
    target_flat, _           = build_flat_df(synth_tgt_omics, tgt_response)

    # ---- Standardize continuous response using source training set stats
    if not is_cat:
        resp_mean = source_flat["response"].mean()
        resp_sd   = source_flat["response"].std(ddof=1)
        if resp_sd > 1e-10:
            source_flat = source_flat.copy()
            target_flat = target_flat.copy()
            source_flat["response"] = (source_flat["response"] - resp_mean) / resp_sd
            target_flat["response"] = (target_flat["response"] - resp_mean) / resp_sd

    # ---- Train / test split on target (100 held-out test samples)
    split_kwargs: dict = dict(train_size=target_size, test_size=100)
    if is_cat:
        split_kwargs["stratify"] = target_flat["response"]
    try:
        target_train, target_eval = train_test_split(target_flat, **split_kwargs)
    except ValueError as e:
        print(f"Train/test split failed for idx={idx}, rep={i}: {e}")
        continue

    if is_cat and target_train["response"].nunique() < 2:
        print(f"  Skipping idx={idx} rep={i}: single class in target train")
        continue

    # ---- Wrap into DatasetContainer
    try:
        datasets = DatasetContainer(
            source_data=source_flat,
            target_data=target_train,
            target_ensemble_data=None,
            target_test_data=[target_eval],
            view_column_groups=view_groups,
        )
        datasets.set_response_column("response")
    except Exception as e:
        print(f"DatasetContainer failed for idx={idx}, rep={i}: {e}")
        continue

    # ---- Output paths
    mlp_outfile = os.path.join(OUT_DIR, f"mlp_expcond_{idx}_replicate_{i}.csv")
    vae_outfile = os.path.join(OUT_DIR, f"vae_expcond_{idx}_replicate_{i}.csv")
    rf_outfile  = os.path.join(OUT_DIR, f"rf_expcond_{idx}_replicate_{i}.csv")

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
        df["shared_var_ratio"]       = shared_var_ratio
        df["modality"]               = modality
        df["alpha"]                  = ALPHA
        df["n_shared_factors"]       = N_SHARED_FACTORS
        return df

    # ---- Fit MLP
    torch.manual_seed(rep_seed)
    try:
        mlp_out, _, _ = fit_dl_model(datasets, "mult_mlp", device("cpu"), param_grid)
        annotate(mlp_out).to_csv(mlp_outfile, index=False)
    except Exception as e:
        print(f"MLP failed for {mlp_outfile}: {e}")

    # ---- Fit VAE
    torch.manual_seed(rep_seed)
    try:
        vae_out, _, _ = fit_dl_model(datasets, "mult_vae", device("cpu"), param_grid)
        annotate(vae_out).to_csv(vae_outfile, index=False)
    except Exception as e:
        print(f"VAE failed for {vae_outfile}: {e}")

    # ---- Fit RF
    random.seed(rep_seed)
    try:
        rf_out, _ = fit_rf_model(datasets)
        annotate(rf_out).to_csv(rf_outfile, index=False)
    except Exception as e:
        print(f"RF failed for {rf_outfile}: {e}")
