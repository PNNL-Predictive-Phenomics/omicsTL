"""
Targeted RF-only rerun for conditions that produced zero RF files
in the full HPC run (all: categorical, alpha=1.0, target_size=100).

Conditions (0-based idx): 89, 92, 182, 188

Run from the devcontainer terminal:
    python /workspaces/timed-hpc/simulation_study_updated/local_rf_rerun.py
"""
import os
import random
import itertools
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

from omicstl.simulation_utils.data_utils import DatasetContainer
from omicstl.simulation_utils.model_utils import fit_rf_model
from omicstl.simulation_utils.data_generation import generate_synth_data_pca, generate_pca_response

WORKSPACE  = "/workspaces/timed-hpc"
DATA_DIR   = os.path.join(WORKSPACE, "docs", "data")
OUT_DIR    = os.path.join(WORKSPACE, "simulation_study_updated", "test_results")
SOURCE_CSV = os.path.join(DATA_DIR, "source_dset.csv")
TARGET_CSV = os.path.join(DATA_DIR, "target_transfer.csv")

TARGET_CONDITIONS = [89, 92, 182, 188]
N_REPS = 10
_CAT_GAMMA = 3


def _run_replicate(
    idx, i,
    source_size, target_size, response_fn_option, response_fn_complexity, snr, alpha,
    source_csv, target_csv, out_dir,
):
    torch.set_num_threads(1)

    rep_seed = int(idx) * 1000 + i
    np.random.seed(rep_seed)
    random.seed(rep_seed)
    torch.manual_seed(rep_seed)

    log = [f"  rep {i} (seed={rep_seed})"]

    base_source_data = pd.read_csv(source_csv).set_index("SampleID").drop("Resp", axis=1)
    base_target_data = pd.read_csv(target_csv).set_index("SampleID").drop("Resp", axis=1)

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
        return "\n".join(log) + f" -> PCA failed: {e}"

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

    if not is_cat:
        resp_mean = source_synth_data["response"].mean()
        resp_sd   = source_synth_data["response"].std(ddof=1)
        if resp_sd > 1e-10:
            source_synth_data = source_synth_data.copy()
            target_synth_data = target_synth_data.copy()
            source_synth_data["response"] = (source_synth_data["response"] - resp_mean) / resp_sd
            target_synth_data["response"] = (target_synth_data["response"] - resp_mean) / resp_sd

    split_kwargs: dict = dict(train_size=target_size, test_size=100)
    if is_cat:
        split_kwargs["stratify"] = target_synth_data["response"]
    try:
        target_synth_train, target_synth_eval = train_test_split(
            target_synth_data, **split_kwargs
        )
    except ValueError as e:
        return "\n".join(log) + f" -> split failed: {e}"

    if is_cat and target_synth_train["response"].nunique() < 2:
        return "\n".join(log) + " -> skipped (single class in target train)"

    try:
        datasets_sim = DatasetContainer(
            source_data=source_synth_data,
            target_data=target_synth_train,
            target_ensemble_data=None,
            target_test_data=[target_synth_eval],
        )
        datasets_sim.set_response_column("response")
    except Exception as e:
        return "\n".join(log) + f" -> DatasetContainer failed: {e}"

    rf_outfile = os.path.join(out_dir, f"rf_expcond_{idx}_replicate_{i}.csv")
    random.seed(rep_seed)
    try:
        rf_out, _ = fit_rf_model(datasets_sim)
        df = rf_out.copy()
        df["exp_idx"]                = idx
        df["replicate"]              = i
        df["rep_seed"]               = rep_seed
        df["source_size"]            = source_size
        df["target_size"]            = target_size
        df["response_fn_option"]     = response_fn_option
        df["response_fn_complexity"] = response_fn_complexity
        df["snr"]                    = snr
        df["alpha"]                  = alpha
        df.to_csv(rf_outfile, index=False)
        return "\n".join(log) + f" -> RF saved"
    except Exception as e:
        return "\n".join(log) + f" -> RF failed: {e}"


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    source_sizes             = [100, 250, 500]
    target_sizes             = [10, 25, 50, 100]
    response_fn_options      = ["cont", "cat"]
    response_fn_complexities = ["linear", "nonlinear"]
    snr_values               = [1, 5]
    alpha_values             = [0.0, 0.5, 1.0]

    exp_conditions_df = pd.DataFrame(
        list(itertools.product(
            source_sizes, target_sizes, response_fn_options,
            response_fn_complexities, snr_values, alpha_values,
        )),
        columns=[
            "source_size", "target_size", "response_fn_option",
            "response_fn_complexity", "snr", "alpha",
        ],
    )

    tasks = [
        (
            idx, i,
            exp_conditions_df.loc[idx, "source_size"].item(),
            exp_conditions_df.loc[idx, "target_size"].item(),
            exp_conditions_df.loc[idx, "response_fn_option"],
            exp_conditions_df.loc[idx, "response_fn_complexity"],
            exp_conditions_df.loc[idx, "snr"].item(),
            exp_conditions_df.loc[idx, "alpha"].item(),
            SOURCE_CSV, TARGET_CSV, OUT_DIR,
        )
        for idx in TARGET_CONDITIONS
        for i in range(N_REPS)
    ]

    n_workers = min(len(tasks), max(1, multiprocessing.cpu_count() - 2))
    print(f"Running {len(tasks)} tasks ({len(TARGET_CONDITIONS)} conditions x {N_REPS} reps) "
          f"across {n_workers} workers\n")
    for idx in TARGET_CONDITIONS:
        row = exp_conditions_df.loc[idx]
        print(f"  condition {idx}: source={row.source_size}, target={row.target_size}, "
              f"response={row.response_fn_option}, complexity={row.response_fn_complexity}, "
              f"snr={row.snr}, alpha={row.alpha}")
    print()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_run_replicate, *t): (t[0], t[1]) for t in tasks}
        for future in as_completed(futures):
            idx, i = futures[future]
            print(f"condition {idx} rep {i}: {future.result()}")

    print("\nDone.")
