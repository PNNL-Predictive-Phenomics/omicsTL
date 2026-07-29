"""Local parallel smoke-test for the multi-omics simulation.

Runs a small representative subset of conditions using Python's
ProcessPoolExecutor so you can verify the full pipeline (data loading,
latent-factor generation, response generation, MLP/VAE/RF fitting,
output CSVs) before pushing to HPC.

Usage:
    python test_multiomics_local.py [--nreps N] [--workers W]

Defaults: nreps=2, workers=4
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent
DATA_DIR    = REPO_ROOT / "docs" / "data"
OUT_DIR     = Path(__file__).resolve().parent / "test_results"   # git-untracked
SCRIPT      = Path(__file__).resolve().parent / "multiomics_simulation_study.py"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Conditions to test ─────────────────────────────────────────────────────
# Grid order: source_size × target_size × response × complexity × snr
#             × shared_var_ratio × modality
# (modality is the innermost / fastest-changing axis)
#
# 1-based task IDs (as SLURM would pass them):
TEST_CONDITIONS = [
    1,   # prot,  cont, linear, snr=1, svr=0.3, src=250, tgt=10  — single-omic baseline
    3,   # multi, cont, linear, snr=1, svr=0.3, src=250, tgt=10  — multi-omics continuous
    27,  # multi, cat,  linear, snr=1, svr=0.3, src=250, tgt=10  — multi-omics categorical
]


def run_condition(task_id: int, nreps: int) -> tuple[int, str, int]:
    """Run one condition in a subprocess and return (task_id, status, returncode)."""
    env = os.environ.copy()
    env["MULTIOMICS_DATA_DIR"] = str(DATA_DIR)
    env["MULTIOMICS_OUT_DIR"]  = str(OUT_DIR)

    cmd = [sys.executable, str(SCRIPT), str(task_id), str(nreps), "1"]
    log_path = OUT_DIR / f"local_task_{task_id}.log"

    with open(log_path, "w") as log:
        result = subprocess.run(
            cmd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    status = "OK" if result.returncode == 0 else "FAILED"
    return task_id, status, result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nreps",   type=int, default=2,  help="Replicates per condition")
    parser.add_argument("--workers", type=int, default=4,  help="Parallel workers")
    args = parser.parse_args()

    print(f"Local multi-omics test")
    print(f"  Script  : {SCRIPT}")
    print(f"  Data    : {DATA_DIR}")
    print(f"  Output  : {OUT_DIR}")
    print(f"  nreps   : {args.nreps}")
    print(f"  workers : {args.workers}")
    print(f"  Tasks   : {TEST_CONDITIONS}")
    print()

    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_condition, tid, args.nreps): tid
            for tid in TEST_CONDITIONS
        }
        for future in as_completed(futures):
            task_id, status, rc = future.result()
            results[task_id] = (status, rc)
            print(f"  task {task_id:>3d}  {status}  (rc={rc})")

    print()
    n_ok     = sum(1 for s, _ in results.values() if s == "OK")
    n_failed = len(results) - n_ok
    print(f"Done: {n_ok}/{len(results)} passed, {n_failed} failed")

    if n_failed:
        print("\nFailed tasks — check logs in docs/data/test_results/:")
        for tid, (status, rc) in sorted(results.items()):
            if status != "OK":
                print(f"  task {tid}: local_task_{tid}.log")
        sys.exit(1)

    # ── Quick output check ────────────────────────────────────────────────
    print("\nOutput files written:")
    csvs = sorted(OUT_DIR.glob("*.csv"))
    for f in csvs:
        rows = sum(1 for _ in open(f)) - 1
        print(f"  {f.name:<50s}  {rows} rows")

    if not csvs:
        print("  (none — something went wrong)")
        sys.exit(1)


if __name__ == "__main__":
    main()
