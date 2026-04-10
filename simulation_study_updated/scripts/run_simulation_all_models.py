#!/usr/bin/env python3
"""
Run omicstl simulation study (MLP + VAE + RF) over a grid of conditions.

Supports:
- Single condition or many conditions (idx list) or all conditions
- Replicates per condition
- Multiple RNG seeds (data-gen seeds + model seeds)
- Local parallelization (multiprocessing)
- SLURM job-array friendly mode (one condition per task)

Output:
Creates CSVs per (model, idx, rep, seed) and also writes an optional combined CSV.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import random
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch import device as torch_device

from omicstl.simulation_utils.data_utils import DatasetContainer
from omicstl.simulation_utils.data_generation import response_function, generate_synth_data
from omicstl.simulation_utils.model_utils import fit_dl_model, fit_rf_model
from omicstl.transfer_forest import load_r_functions


# -----------------------------
# Grid definition (matches your HPC script)
# -----------------------------
SOURCE_SIZES = [25, 50, 100, 200]
TARGET_SIZES = [5, 10, 25, 192]
RESPONSE_FN_OPTIONS = ["cont", "cat"]
RESPONSE_FN_COMPLEXITIES = ["linear", "nonlinear"]
SNRS_SOURCE = [0.1, 0.5, 1, 2]
SNRS_TARGET = [0.1, 0.5, 1, 2]
FEATURE_RATIOS = [0.5, 1, 2]

DEFAULT_PARAM_GRID = {
    "dropout": [0.25, 0.5],
    "n_latent_dims": [2],
    "hidden_dim_base": [6],
    "lr": [0.01, 0.001],
    "source_epochs": [1000],
    "target_epochs": [1000],
    "freeze": ["none"],
    "weight_decay": [1e-4, 1e-2],
    "gamma": [1, 2, 3],
}


@dataclass(frozen=True)
class Condition:
    idx: int
    source_size: int
    target_size: int
    response_fn_option: str
    response_fn_complexity: str
    snr_source: float
    snr_target: float
    feature_ratio: float

    def tag(self) -> str:
        return (
            f"idx{self.idx}"
            f"_src{self.source_size}"
            f"_tgt{self.target_size}"
            f"_{self.response_fn_option}"
            f"_{self.response_fn_complexity}"
            f"_snrs{self.snr_source}"
            f"_snrt{self.snr_target}"
            f"_fr{self.feature_ratio}"
        )


def build_conditions_df() -> pd.DataFrame:
    exp_conditions = list(
        itertools.product(
            SOURCE_SIZES,
            TARGET_SIZES,
            RESPONSE_FN_OPTIONS,
            RESPONSE_FN_COMPLEXITIES,
            SNRS_SOURCE,
            SNRS_TARGET,
            FEATURE_RATIOS,
        )
    )
    return pd.DataFrame(
        exp_conditions,
        columns=[
            "source_size",
            "target_size",
            "response_fn_options",
            "response_fn_complexity",
            "snr_source",
            "snr_target",
            "feature_ratio",
        ],
    )


def get_condition(exp_df: pd.DataFrame, idx: int) -> Condition:
    if not (0 <= idx < len(exp_df)):
        raise ValueError(f"IDX={idx} out of range. Must be 0..{len(exp_df)-1}")
    row = exp_df.loc[idx]
    return Condition(
        idx=idx,
        source_size=int(row["source_size"]),
        target_size=int(row["target_size"]),
        response_fn_option=str(row["response_fn_options"]),
        response_fn_complexity=str(row["response_fn_complexity"]),
        snr_source=float(row["snr_source"]),
        snr_target=float(row["snr_target"]),
        feature_ratio=float(row["feature_ratio"]),
    )


def parse_int_list(s: str) -> list[int]:
    """
    Accepts formats:
      "1,2,3"
      "1-5"
      "1-5,10,12-14"
    """
    out: list[int] = []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            a_i, b_i = int(a), int(b)
            step = 1 if b_i >= a_i else -1
            out.extend(list(range(a_i, b_i + step, step)))
        else:
            out.append(int(p))
    return out


def set_all_seeds(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # safe even on CPU-only
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_response_fn(complexity: str):
    if complexity == "nonlinear":
        return response_function("tanh(df[, 2]) + df[, 1] * df[, ncol(df)] ^ 2")
    if complexity == "linear":
        return response_function("df[, 2] + df[, 1] * df[, ncol(df)]")
    raise ValueError(f"Unknown response_fn_complexity: {complexity}")


def build_datasets_for_condition(
    cond: Condition,
    base_source_data: pd.DataFrame,
    base_target_transfer_data: pd.DataFrame,
    data_seed: int,
) -> DatasetContainer:
    """
    Generates synthetic source + target datasets for one condition and one seed.
    """
    # seed the data generation (python + numpy are what most gen uses)
    random.seed(data_seed)
    np.random.seed(data_seed)

    response_fn = make_response_fn(cond.response_fn_complexity)
    num_features = math.floor(cond.feature_ratio * max(cond.source_size, cond.target_size))

    if cond.response_fn_option == "cont":
        source_synth_data, source_lc_info, _ = generate_synth_data(
            data=base_source_data,
            num_features=num_features,
            num_samples=cond.source_size,
            response_fn=response_fn,
            snr=cond.snr_source,
        )
        target_synth_data, _, _ = generate_synth_data(
            data=base_target_transfer_data,
            num_features=num_features,
            num_samples=cond.target_size + 100,
            response_fn=response_fn,
            snr=cond.snr_target,
            prior_lc_info=source_lc_info,
        )

        source_synth_data.rename(columns={source_synth_data.columns[0]: "response"}, inplace=True)
        target_synth_data.rename(columns={target_synth_data.columns[0]: "response"}, inplace=True)

        target_synth_train, target_synth_eval = train_test_split(
            target_synth_data, train_size=cond.target_size, test_size=100
        )

    elif cond.response_fn_option == "cat":
        source_synth_data, source_lc_info, _ = generate_synth_data(
            data=base_source_data,
            num_features=num_features,
            num_samples=cond.source_size,
            response_fn=response_fn,
            snr=cond.snr_source,
            response_parameters={"ncats": 2, "quantile": "quantile"},
        )
        target_synth_data, _, _ = generate_synth_data(
            data=base_target_transfer_data,
            num_features=num_features,
            num_samples=cond.target_size + 100,
            response_fn=response_fn,
            snr=cond.snr_target,
            prior_lc_info=source_lc_info,
            response_parameters={"ncats": 2, "quantile": "quantile"},
        )

        source_synth_data.rename(columns={source_synth_data.columns[0]: "response"}, inplace=True)
        target_synth_data.rename(columns={target_synth_data.columns[0]: "response"}, inplace=True)

        source_synth_data = source_synth_data.astype({"response": np.int64})
        target_synth_data = target_synth_data.astype({"response": np.int64})

        target_synth_train, target_synth_eval = train_test_split(
            target_synth_data,
            train_size=cond.target_size,
            test_size=100,
            stratify=target_synth_data["response"],
        )
    else:
        raise ValueError(f"Unknown response_fn_option: {cond.response_fn_option}")

    ds = DatasetContainer(
        source_data=source_synth_data,
        target_data=target_synth_train,
        target_ensemble_data=None,
        target_test_data=[target_synth_eval],
    )
    ds.set_response_column("response")
    return ds


def add_meta(
    out_df: pd.DataFrame,
    model_name: str,
    cond: Condition,
    rep: int,
    data_seed: int,
    model_seed: int,
) -> pd.DataFrame:
    df = out_df.copy()
    df["model"] = model_name
    df["idx"] = cond.idx
    df["rep"] = rep
    df["data_seed"] = data_seed
    df["model_seed"] = model_seed
    df["source_size"] = cond.source_size
    df["target_size"] = cond.target_size
    df["snr_source"] = cond.snr_source
    df["snr_target"] = cond.snr_target
    df["feature_ratio"] = cond.feature_ratio
    df["response_fn_option"] = cond.response_fn_option
    df["response_fn_complexity"] = cond.response_fn_complexity
    return df


def run_one(
    cond: Condition,
    rep: int,
    data_seed: int,
    model_seed: int,
    base_source_data: pd.DataFrame,
    base_target_transfer_data: pd.DataFrame,
    out_dir: Path,
    param_grid: dict | None,
    do_mlp: bool,
    do_vae: bool,
    do_rf: bool,
    torch_det: bool,
    device_str: str,
) -> list[Path]:
    """
    Runs the selected models for one (condition, rep, seed).
    Returns list of written output files.
    """
    written: list[Path] = []

    # build data with data_seed (keeps data variation separate from model variation)
    ds = build_datasets_for_condition(
        cond=cond,
        base_source_data=base_source_data,
        base_target_transfer_data=base_target_transfer_data,
        data_seed=data_seed,
    )

    # set model seeds right before fitting models
    set_all_seeds(model_seed, deterministic=torch_det)
    dev = torch_device(device_str)

    stem = f"{cond.tag()}_rep{rep}_dataseed{data_seed}_modelseed{model_seed}"

    if do_mlp:
        mlp_out = out_dir / f"mlp_{stem}.csv"
        try:
            out_df, _, _ = fit_dl_model(ds, "mult_mlp", dev, param_grid)
            out_df = add_meta(out_df, "mlp", cond, rep, data_seed, model_seed)
            out_df.to_csv(mlp_out, index=False)
            print(f"[OK] MLP -> {mlp_out}")
            written.append(mlp_out)
        except Exception:
            print(f"[FAIL] MLP {stem}")
            print(traceback.format_exc())

    if do_vae:
        vae_out = out_dir / f"vae_{stem}.csv"
        try:
            out_df, _, _ = fit_dl_model(ds, "mult_vae", dev, param_grid)
            out_df = add_meta(out_df, "vae", cond, rep, data_seed, model_seed)
            out_df.to_csv(vae_out, index=False)
            print(f"[OK] VAE -> {vae_out}")
            written.append(vae_out)
        except Exception:
            print(f"[FAIL] VAE {stem}")
            print(traceback.format_exc())

    if do_rf:
        rf_out = out_dir / f"rf_{stem}.csv"
        try:
            out_df, _ = fit_rf_model(ds)
            out_df = add_meta(out_df, "rf", cond, rep, data_seed, model_seed)
            out_df.to_csv(rf_out, index=False)
            print(f"[OK] RF  -> {rf_out}")
            written.append(rf_out)
        except Exception:
            print(f"[FAIL] RF {stem}")
            print(traceback.format_exc())

    return written


# -----------------------------
# Multiprocessing (spawn-safe)
# -----------------------------
_WORKER_CTX: dict | None = None
_R_LOADED: bool = False


def _init_worker(ctx: dict) -> None:
    """Initializer: runs once per child process."""
    global _WORKER_CTX, _R_LOADED
    _WORKER_CTX = ctx
    _R_LOADED = False


def _worker_task(tup) -> list[Path]:
    """Top-level worker (pickle-safe)."""
    global _WORKER_CTX, _R_LOADED
    if _WORKER_CTX is None:
        raise RuntimeError("Worker context not initialized (missing initializer).")

    cond, rep, ds, ms = tup

    # Load R funcs once per child if RF is requested
    if _WORKER_CTX["do_rf"] and not _R_LOADED:
        load_r_functions()
        _R_LOADED = True

    try:
        return run_one(
            cond=cond,
            rep=rep,
            data_seed=ds,
            model_seed=ms,
            base_source_data=_WORKER_CTX["base_source_data"],
            base_target_transfer_data=_WORKER_CTX["base_target_transfer_data"],
            out_dir=_WORKER_CTX["out_dir"],
            param_grid=_WORKER_CTX["param_grid"],
            do_mlp=_WORKER_CTX["do_mlp"],
            do_vae=_WORKER_CTX["do_vae"],
            do_rf=_WORKER_CTX["do_rf"],
            torch_det=_WORKER_CTX["torch_det"],
            device_str=_WORKER_CTX["device_str"],
        )
    except Exception:
        print(f"[FAIL] worker crashed: {cond.tag()} rep={rep} dataseed={ds} modelseed={ms}")
        print(traceback.format_exc())
        return []


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser()

    # Paths
    p.add_argument(
        "--repo-dir",
        type=str,
        default=None,
        help="Repo root. Default: infer from this script location.",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing source_dset.csv and target_transfer.csv.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for results CSVs.",
    )

    # Condition selection
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--idx", type=int, default=None, help="Run a single condition index (0-based).")
    g.add_argument(
        "--idx-list",
        type=str,
        default=None,
        help='Run multiple indices, e.g. "0,150,200,300,383" or "0-100".',
    )
    g.add_argument("--all", action="store_true", help="Run all conditions (0..N-1).")

    # Replicates & seeds
    p.add_argument("--nreps", type=int, default=1, help="Replicates per condition.")
    p.add_argument(
        "--data-seeds",
        type=str,
        default="123",
        help='Comma/range list. Example: "11,21,31" or "11-15".',
    )
    p.add_argument(
        "--model-seeds",
        type=str,
        default="42",
        help='Comma/range list. Example: "42,43,44" or "42-46".',
    )

    # Models
    p.add_argument(
        "--models",
        type=str,
        default="mlp,vae,rf",
        help='Comma list from {mlp,vae,rf}. Default: all.',
    )

    # DL tuning grid
    p.add_argument(
        "--no-tuning",
        action="store_true",
        help="Disable tuning (NOT recommended with current fit_dl_model; may error).",
    )
    p.add_argument("--device", type=str, default="cpu", help='Torch device string, e.g. "cpu" or "cuda:0".')
    p.add_argument(
        "--torch-deterministic",
        action="store_true",
        help="Force torch deterministic mode (slower).",
    )

    # Local parallelization
    p.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Local parallel workers. Use 1 for serial. For SLURM, prefer job arrays instead.",
    )

    # SLURM array convenience
    p.add_argument(
        "--slurm-array-task",
        type=int,
        default=None,
        help="If set, runs IDX = (task_id-1). If not set, will also honor SLURM_ARRAY_TASK_ID env var.",
    )

    # Combine outputs
    p.add_argument(
        "--write-combined",
        action="store_true",
        help="Also write combined_results.csv in out-dir for what this run produced.",
    )

    args = p.parse_args(argv)

    # Infer repo/data/out dirs
    if args.repo_dir is None:
        script_path = Path(__file__).resolve()
        # Adjust this if you place the script elsewhere in the repo
        repo_dir = next(p for p in script_path.parents if (p / "pyproject.toml").exists())
    else:
        repo_dir = Path(args.repo_dir).resolve()

    data_dir = Path(args.data_dir).resolve() if args.data_dir else (repo_dir / "docs" / "data")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (repo_dir / "simulation_study_updated" / "results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load base datasets
    src_path = data_dir / "source_dset.csv"
    tgt_path = data_dir / "target_transfer.csv"
    if not src_path.exists() or not tgt_path.exists():
        raise FileNotFoundError(f"Missing required files in {data_dir}: source_dset.csv / target_transfer.csv")

    base_source_data = pd.read_csv(src_path).set_index("SampleID")
    base_source_data = base_source_data.drop("Resp", axis=1, errors="ignore")

    base_target_transfer_data = pd.read_csv(tgt_path).set_index("SampleID")
    base_target_transfer_data = base_target_transfer_data.drop("Resp", axis=1, errors="ignore")

    exp_df = build_conditions_df()

    # condition indices (honor SLURM_ARRAY_TASK_ID if present)
    task_id = args.slurm_array_task
    if task_id is None:
        env_task = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_task is not None:
            task_id = int(env_task)

    if task_id is not None:
        idxs = [int(task_id) - 1]
    elif args.idx is not None:
        idxs = [args.idx]
    elif args.idx_list is not None:
        idxs = parse_int_list(args.idx_list)
    else:
        idxs = list(range(len(exp_df)))  # all

    # validate idxs early
    for i in idxs:
        if not (0 <= i < len(exp_df)):
            raise ValueError(f"IDX={i} out of range. Must be 0..{len(exp_df)-1}")

    # model selection
    models = {m.strip().lower() for m in args.models.split(",") if m.strip()}
    do_mlp = "mlp" in models
    do_vae = "vae" in models
    do_rf = "rf" in models

    if not (do_mlp or do_vae or do_rf):
        raise ValueError("No valid models selected. Use --models mlp,vae,rf (any subset).")

    # seeds
    data_seeds = parse_int_list(args.data_seeds)
    model_seeds = parse_int_list(args.model_seeds)

    # param_grid
    if args.no_tuning:
        # WARNING: fit_dl_model in some repos assumes a tuning path.
        param_grid = None
    else:
        param_grid = DEFAULT_PARAM_GRID

    # build all tasks
    tasks = []
    for idx in idxs:
        cond = get_condition(exp_df, idx)
        for rep in range(1, args.nreps + 1):  # 1-based reps
            for ds in data_seeds:
                for ms in model_seeds:
                    tasks.append((cond, rep, ds, ms))

    print(
        f"Will run: {len(idxs)} conditions × {args.nreps} reps × {len(data_seeds)} data_seeds × {len(model_seeds)} model_seeds"
    )
    print(f"Total tasks (per-model inside each): {len(tasks)}")
    if len(idxs) == 1:
        print(f"Condition {idxs[0]}: {get_condition(exp_df, idxs[0])}")

    all_written: list[Path] = []

    if args.n_jobs <= 1:
        # Serial path: if RF is enabled, load R funcs once here
        if do_rf:
            load_r_functions()

        for cond, rep, ds, ms in tasks:
            all_written.extend(
                run_one(
                    cond=cond,
                    rep=rep,
                    data_seed=ds,
                    model_seed=ms,
                    base_source_data=base_source_data,
                    base_target_transfer_data=base_target_transfer_data,
                    out_dir=out_dir,
                    param_grid=param_grid,
                    do_mlp=do_mlp,
                    do_vae=do_vae,
                    do_rf=do_rf,
                    torch_det=args.torch_deterministic,
                    device_str=args.device,
                )
            )
    else:
        # Parallel path (spawn-safe)
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        worker_ctx = {
            "base_source_data": base_source_data,
            "base_target_transfer_data": base_target_transfer_data,
            "out_dir": out_dir,
            "param_grid": param_grid,
            "do_mlp": do_mlp,
            "do_vae": do_vae,
            "do_rf": do_rf,
            "torch_det": args.torch_deterministic,
            "device_str": args.device,
        }

        with ctx.Pool(
            processes=args.n_jobs,
            initializer=_init_worker,
            initargs=(worker_ctx,),
        ) as pool:
            for written_list in pool.imap_unordered(_worker_task, tasks, chunksize=1):
                all_written.extend(written_list)

    if args.write_combined and all_written:
        dfs = []
        for f in all_written:
            try:
                df = pd.read_csv(f)
                df["file"] = f.name
                dfs.append(df)
            except Exception:
                pass
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            combined_path = out_dir / "combined_results.csv"
            combined.to_csv(combined_path, index=False)
            print(f"[OK] Wrote combined results -> {combined_path}")

    print(f"Done. Wrote {len(all_written)} result files into: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())