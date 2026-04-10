#!/usr/bin/env python3
"""
Run omicstl simulation study (MLP + VAE + RF) over a grid of conditions.

Speed fixes included (the “six things”):
1) REP now actually changes randomness (rep is folded into data/model seeds).
2) Data are generated ONCE per (condition, rep, data_seed) and reused across ALL model_seeds.
3) In multiprocessing (spawn-safe), workers load base CSVs once (we pass file paths, not big DataFrames).
4) DL tuning/epochs budget reduced by default + optional random-subsampling of the tuning grid.
5) Thread oversubscription mitigated (OMP/MKL/OPENBLAS/NUMEXPR + torch threads set to 1).
6) I/O reduced by default: write ONE results file per (condition, rep, data_seed) “unit”
   (you can still split into many files if you want).

Notes:
- RF reproducibility: we try to set an R seed (if rpy2 is available) AND we try to pass random_state
  into fit_rf_model if it supports it. Otherwise, we at least reseed python/numpy before RF.
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
from typing import Any, Sequence

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

# Default is made *much* cheaper than 1000/1000 + full grid.
DEFAULT_PARAM_GRID = {
    "dropout": [0.25, 0.5],
    "n_latent_dims": [2],
    "hidden_dim_base": [6],
    "lr": [0.01, 0.001],
    "source_epochs": [300],  # was 1000
    "target_epochs": [300],  # was 1000
    "freeze": ["none"],
    "weight_decay": [1e-4, 1e-2],
    "gamma": [1, 2, 3],
}


# -----------------------------
# Determinism / thread control
# -----------------------------
def configure_threading(single_thread: bool = True) -> None:
    """Prevent oversubscription when running multiple processes."""
    if not single_thread:
        return
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def set_all_seeds(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def try_set_r_seed(seed: int) -> None:
    """
    Best-effort: set R RNG seed for the RF/transfer-forest stack.
    If rpy2 isn't available in your env, we just skip.
    """
    try:
        import rpy2.robjects as ro  # type: ignore

        ro.r(f"set.seed({int(seed)})")
    except Exception:
        # silently ignore; reproducibility still improved via other seeds
        return


# -----------------------------
# Helpers
# -----------------------------
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


# Cache response functions so we don't rebuild the R expression every run
_RESPONSE_FN_CACHE: dict[str, Any] = {}


def make_response_fn(complexity: str):
    if complexity in _RESPONSE_FN_CACHE:
        return _RESPONSE_FN_CACHE[complexity]

    if complexity == "nonlinear":
        fn = response_function("tanh(df[, 2]) + df[, 1] * df[, ncol(df)] ^ 2")
    elif complexity == "linear":
        fn = response_function("df[, 2] + df[, 1] * df[, ncol(df)]")
    else:
        raise ValueError(f"Unknown response_fn_complexity: {complexity}")

    _RESPONSE_FN_CACHE[complexity] = fn
    return fn


def build_datasets_for_condition(
    cond: Condition,
    base_source_data: pd.DataFrame,
    base_target_transfer_data: pd.DataFrame,
    effective_data_seed: int,
) -> DatasetContainer:
    """
    Generates synthetic source + target datasets for one condition and one effective seed.
    """
    random.seed(effective_data_seed)
    np.random.seed(effective_data_seed)

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
    effective_data_seed: int,
    effective_model_seed: int,
) -> pd.DataFrame:
    df = out_df.copy()
    df["model"] = model_name
    df["idx"] = cond.idx
    df["rep"] = rep

    # original seeds from CLI
    df["data_seed"] = data_seed
    df["model_seed"] = model_seed

    # actual seeds used (rep folded in)
    df["effective_data_seed"] = effective_data_seed
    df["effective_model_seed"] = effective_model_seed

    df["source_size"] = cond.source_size
    df["target_size"] = cond.target_size
    df["snr_source"] = cond.snr_source
    df["snr_target"] = cond.snr_target
    df["feature_ratio"] = cond.feature_ratio
    df["response_fn_option"] = cond.response_fn_option
    df["response_fn_complexity"] = cond.response_fn_complexity
    return df


def product_dict(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Expand a param grid dict into a list of param dicts."""
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    out = []
    for combo in itertools.product(*vals):
        out.append({k: v for k, v in zip(keys, combo)})
    return out


def subsample_param_grid(param_grid: dict[str, list[Any]], k: int, seed: int) -> dict[str, list[Any]] | None:
    """
    Deterministically subsample the FULL cartesian grid to k configs, then return a grid-like dict
    that fit_dl_model can still accept (if it expects a grid).
    """
    if param_grid is None:
        return None
    full = product_dict(param_grid)
    if k <= 0 or k >= len(full):
        return param_grid

    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(full), size=k, replace=False)
    sampled = [full[i] for i in idxs]

    # Convert sampled list-of-dicts back into "grid" form with unique values per key
    new_grid: dict[str, list[Any]] = {}
    for key in param_grid.keys():
        new_grid[key] = sorted({d[key] for d in sampled}, key=lambda x: str(x))
    return new_grid


def write_results(df: pd.DataFrame, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    elif fmt == "csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unknown format: {fmt}")


# -----------------------------
# Work unit: generate data ONCE, then run all model_seeds
# -----------------------------
def run_unit(
    cond: Condition,
    rep: int,
    data_seed: int,
    model_seeds: list[int],
    base_source_data: pd.DataFrame,
    base_target_transfer_data: pd.DataFrame,
    out_dir: Path,
    param_grid: dict[str, list[Any]] | None,
    do_mlp: bool,
    do_vae: bool,
    do_rf: bool,
    torch_det: bool,
    device_str: str,
    out_format: str,
    split_files: bool,
) -> list[Path]:
    """
    Runs one unit of work:
      - Build datasets once using effective_data_seed (data_seed + rep)
      - For each model_seed: fit selected models
      - Write either one combined file (default) OR split files per model/seed
    """
    written: list[Path] = []

    # Make REP matter
    effective_data_seed = data_seed * 1000 + rep

    ds = build_datasets_for_condition(
        cond=cond,
        base_source_data=base_source_data,
        base_target_transfer_data=base_target_transfer_data,
        effective_data_seed=effective_data_seed,
    )

    dev = torch_device(device_str)

    results_rows: list[pd.DataFrame] = []

    # If RF is enabled, ensure R funcs are loaded and ready in this process
    if do_rf:
        load_r_functions()

    for ms in model_seeds:
        effective_model_seed = ms * 1000 + rep

        # Seed right before each model run
        set_all_seeds(effective_model_seed, deterministic=torch_det)

        # Also seed R (best-effort) for RF stack
        if do_rf:
            try_set_r_seed(effective_model_seed)

        stem = f"{cond.tag()}_rep{rep}_dataseed{data_seed}_modelseed{ms}"

        if do_mlp:
            try:
                out_df, _, _ = fit_dl_model(ds, "mult_mlp", dev, param_grid)
                out_df = add_meta(
                    out_df, "mlp", cond, rep, data_seed, ms, effective_data_seed, effective_model_seed
                )
                if split_files:
                    out_path = out_dir / f"mlp_{stem}.{ 'parquet' if out_format=='parquet' else 'csv' }"
                    write_results(out_df, out_path, out_format)
                    written.append(out_path)
                    print(f"[OK] MLP -> {out_path}")
                else:
                    results_rows.append(out_df)
            except Exception:
                print(f"[FAIL] MLP {stem}")
                print(traceback.format_exc())

        if do_vae:
            try:
                out_df, _, _ = fit_dl_model(ds, "mult_vae", dev, param_grid)
                out_df = add_meta(
                    out_df, "vae", cond, rep, data_seed, ms, effective_data_seed, effective_model_seed
                )
                if split_files:
                    out_path = out_dir / f"vae_{stem}.{ 'parquet' if out_format=='parquet' else 'csv' }"
                    write_results(out_df, out_path, out_format)
                    written.append(out_path)
                    print(f"[OK] VAE -> {out_path}")
                else:
                    results_rows.append(out_df)
            except Exception:
                print(f"[FAIL] VAE {stem}")
                print(traceback.format_exc())

        if do_rf:
            try:
                # Try to pass a seed if fit_rf_model supports it; otherwise fallback.
                try:
                    out_df, _ = fit_rf_model(ds, random_state=int(effective_model_seed))
                except TypeError:
                    # At least reseed python/numpy before RF
                    random.seed(effective_model_seed)
                    np.random.seed(effective_model_seed)
                    out_df, _ = fit_rf_model(ds)

                out_df = add_meta(
                    out_df, "rf", cond, rep, data_seed, ms, effective_data_seed, effective_model_seed
                )
                if split_files:
                    out_path = out_dir / f"rf_{stem}.{ 'parquet' if out_format=='parquet' else 'csv' }"
                    write_results(out_df, out_path, out_format)
                    written.append(out_path)
                    print(f"[OK] RF  -> {out_path}")
                else:
                    results_rows.append(out_df)
            except Exception:
                print(f"[FAIL] RF {stem}")
                print(traceback.format_exc())

    # Default: write ONE file per unit to reduce I/O
    if (not split_files) and results_rows:
        unit_df = pd.concat(results_rows, ignore_index=True)
        unit_stem = f"{cond.tag()}_rep{rep}_dataseed{data_seed}"
        unit_path = out_dir / f"results_{unit_stem}.{ 'parquet' if out_format=='parquet' else 'csv' }"
        write_results(unit_df, unit_path, out_format)
        written.append(unit_path)
        print(f"[OK] UNIT -> {unit_path} (rows={len(unit_df)})")

    return written


# -----------------------------
# Multiprocessing (spawn-safe)
# -----------------------------
_WORKER_CTX: dict[str, Any] | None = None


def _init_worker(ctx: dict[str, Any]) -> None:
    """Initializer: runs once per child process."""
    global _WORKER_CTX

    configure_threading(single_thread=True)

    # Load base datasets ONCE per worker (avoid pickling huge DataFrames)
    src_path = Path(ctx["src_path"])
    tgt_path = Path(ctx["tgt_path"])

    base_source_data = pd.read_csv(src_path).set_index("SampleID").drop("Resp", axis=1, errors="ignore")
    base_target_transfer_data = pd.read_csv(tgt_path).set_index("SampleID").drop("Resp", axis=1, errors="ignore")

    new_ctx = dict(ctx)
    new_ctx["base_source_data"] = base_source_data
    new_ctx["base_target_transfer_data"] = base_target_transfer_data
    _WORKER_CTX = new_ctx


def _worker_task(tup) -> list[Path]:
    """Top-level worker (pickle-safe)."""
    global _WORKER_CTX
    if _WORKER_CTX is None:
        raise RuntimeError("Worker context not initialized (missing initializer).")

    cond, rep, data_seed = tup

    try:
        return run_unit(
            cond=cond,
            rep=rep,
            data_seed=data_seed,
            model_seeds=_WORKER_CTX["model_seeds"],
            base_source_data=_WORKER_CTX["base_source_data"],
            base_target_transfer_data=_WORKER_CTX["base_target_transfer_data"],
            out_dir=_WORKER_CTX["out_dir"],
            param_grid=_WORKER_CTX["param_grid"],
            do_mlp=_WORKER_CTX["do_mlp"],
            do_vae=_WORKER_CTX["do_vae"],
            do_rf=_WORKER_CTX["do_rf"],
            torch_det=_WORKER_CTX["torch_det"],
            device_str=_WORKER_CTX["device_str"],
            out_format=_WORKER_CTX["out_format"],
            split_files=_WORKER_CTX["split_files"],
        )
    except Exception:
        print(f"[FAIL] worker crashed: {cond.tag()} rep={rep} dataseed={data_seed}")
        print(traceback.format_exc())
        return []


# -----------------------------
# Main
# -----------------------------
def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser()

    # Paths
    p.add_argument("--repo-dir", type=str, default=None, help="Repo root. Default: infer from this script location.")
    p.add_argument(
        "--data-dir", type=str, default=None, help="Directory containing source_dset.csv and target_transfer.csv."
    )
    p.add_argument("--out-dir", type=str, default=None, help="Output directory for results files.")

    # Condition selection
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--idx", type=int, default=None, help="Run a single condition index (0-based).")
    g.add_argument("--idx-list", type=str, default=None, help='Run multiple indices, e.g. "0,150,200" or "0-100".')
    g.add_argument("--all", action="store_true", help="Run all conditions (0..N-1).")

    # Replicates & seeds
    p.add_argument("--nreps", type=int, default=1, help="Replicates per condition.")
    p.add_argument("--data-seeds", type=str, default="123", help='Comma/range list. Example: "11,21,31" or "11-15".')
    p.add_argument(
        "--model-seeds", type=str, default="42", help='Comma/range list. Example: "42,43,44" or "42-46".'
    )

    # Models
    p.add_argument("--models", type=str, default="mlp,vae,rf", help='Comma list from {mlp,vae,rf}. Default: all.')

    # DL tuning controls
    p.add_argument("--no-tuning", action="store_true", help="Disable tuning (passes None to fit_dl_model).")
    p.add_argument(
        "--tuning-samples",
        type=int,
        default=8,
        help="Randomly subsample tuning grid to this many configs (0/large => full grid).",
    )
    p.add_argument("--source-epochs", type=int, default=300, help="Override source_epochs in the DL tuning grid.")
    p.add_argument("--target-epochs", type=int, default=300, help="Override target_epochs in the DL tuning grid.")
    p.add_argument("--tuning-seed", type=int, default=0, help="Seed used to subsample the tuning grid deterministically.")

    # Torch
    p.add_argument("--device", type=str, default="cpu", help='Torch device string, e.g. "cpu" or "cuda:0".')
    p.add_argument("--torch-deterministic", action="store_true", help="Force torch deterministic mode (slower).")

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

    # Output options (I/O speed)
    p.add_argument("--out-format", choices=["csv", "parquet"], default="parquet", help="Output format.")
    p.add_argument(
        "--split-files",
        action="store_true",
        help="If set, writes per-model/per-modelseed files (more files, slower). Default writes one file per unit.",
    )

    # Combine outputs from *this invocation* (optional)
    p.add_argument("--write-combined", action="store_true", help="Also write combined_results.(csv|parquet) in out-dir.")

    args = p.parse_args(argv)

    # Thread control (important for speed)
    configure_threading(single_thread=True)

    # Infer repo/data/out dirs
    if args.repo_dir is None:
        script_path = Path(__file__).resolve()
        repo_dir = next(pp for pp in script_path.parents if (pp / "pyproject.toml").exists())
    else:
        repo_dir = Path(args.repo_dir).resolve()

    data_dir = Path(args.data_dir).resolve() if args.data_dir else (repo_dir / "docs" / "data")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (repo_dir / "simulation_study_updated" / "results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Base dataset paths
    src_path = data_dir / "source_dset.csv"
    tgt_path = data_dir / "target_transfer.csv"
    if not src_path.exists() or not tgt_path.exists():
        raise FileNotFoundError(f"Missing required files in {data_dir}: source_dset.csv / target_transfer.csv")

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

    # param_grid (cheaper defaults + optional subsample)
    if args.no_tuning:
        param_grid = None
    else:
        param_grid = dict(DEFAULT_PARAM_GRID)
        param_grid["source_epochs"] = [int(args.source_epochs)]
        param_grid["target_epochs"] = [int(args.target_epochs)]
        param_grid = subsample_param_grid(param_grid, k=int(args.tuning_samples), seed=int(args.tuning_seed))

    # Build units: (cond, rep, data_seed). Each unit reuses data across ALL model_seeds.
    units: list[tuple[Condition, int, int]] = []
    for idx in idxs:
        cond = get_condition(exp_df, idx)
        for rep in range(1, args.nreps + 1):
            for ds in data_seeds:
                units.append((cond, rep, ds))

    print(
        f"Will run: {len(idxs)} conditions × {args.nreps} reps × {len(data_seeds)} data_seeds "
        f"(data generated per unit, reused across {len(model_seeds)} model_seeds)"
    )
    print(f"Total UNITS: {len(units)}")
    print(f"Per unit, model_seeds: {len(model_seeds)}; models: {sorted(models)}")
    if len(idxs) == 1:
        print(f"Condition {idxs[0]}: {get_condition(exp_df, idxs[0])}")

    all_written: list[Path] = []

    if args.n_jobs <= 1:
        # Serial path: load base datasets once
        base_source_data = pd.read_csv(src_path).set_index("SampleID").drop("Resp", axis=1, errors="ignore")
        base_target_transfer_data = pd.read_csv(tgt_path).set_index("SampleID").drop("Resp", axis=1, errors="ignore")

        for cond, rep, ds in units:
            all_written.extend(
                run_unit(
                    cond=cond,
                    rep=rep,
                    data_seed=ds,
                    model_seeds=model_seeds,
                    base_source_data=base_source_data,
                    base_target_transfer_data=base_target_transfer_data,
                    out_dir=out_dir,
                    param_grid=param_grid,
                    do_mlp=do_mlp,
                    do_vae=do_vae,
                    do_rf=do_rf,
                    torch_det=args.torch_deterministic,
                    device_str=args.device,
                    out_format=args.out_format,
                    split_files=args.split_files,
                )
            )
    else:
        # Parallel path: spawn-safe, workers load base datasets ONCE each
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        worker_ctx = {
            "src_path": str(src_path),
            "tgt_path": str(tgt_path),
            "out_dir": out_dir,
            "param_grid": param_grid,
            "do_mlp": do_mlp,
            "do_vae": do_vae,
            "do_rf": do_rf,
            "torch_det": args.torch_deterministic,
            "device_str": args.device,
            "model_seeds": model_seeds,
            "out_format": args.out_format,
            "split_files": args.split_files,
        }

        # chunksize > 1 reduces scheduling overhead
        chunksize = max(1, len(units) // (args.n_jobs * 4)) if len(units) > 0 else 1

        with ctx.Pool(
            processes=args.n_jobs,
            initializer=_init_worker,
            initargs=(worker_ctx,),
        ) as pool:
            for written_list in pool.imap_unordered(_worker_task, units, chunksize=chunksize):
                all_written.extend(written_list)

    # Optional: combine outputs written by THIS invocation
    if args.write_combined and all_written:
        dfs: list[pd.DataFrame] = []
        for f in all_written:
            try:
                if args.out_format == "parquet":
                    df = pd.read_parquet(f)
                else:
                    df = pd.read_csv(f)
                df["file"] = f.name
                dfs.append(df)
            except Exception:
                pass

        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            combined_path = out_dir / f"combined_results.{args.out_format}"
            write_results(combined, combined_path, args.out_format)
            print(f"[OK] Wrote combined results -> {combined_path}")

    print(f"Done. Wrote {len(all_written)} result files into: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())