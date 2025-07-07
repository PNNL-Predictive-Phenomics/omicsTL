#!/usr/bin/env python3
"""
Script to run model fitting on multiple datasets in parallel using multiprocessing.
"""

import argparse
import logging
import multiprocessing as mp
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from torch import device
from tqdm import tqdm

from omicsTL.simulation_utils.data_utils import DatasetContainer, DatasetManager
from omicsTL.simulation_utils.model_utils import fit_dl_model, fit_rf_model
from omicsTL.transfer_forest import load_r_functions

# Create main logger
logger = logging.getLogger(__name__)


def setup_logging(log_file: str | None = None) -> None:
    """Set up logging configuration for all modules."""

    # Replace task ID placeholder when multi-node processing is used
    if log_file is not None and os.getenv("SLURM_ARRAY_TASK_ID") is not None:
        log_file.replace("%a", os.getenv("SLURM_ARRAY_TASK_ID"))

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    for name in logging.root.manager.loggerDict:
        if name.startswith("omicsTL."):
            module_logger = logging.getLogger(name)
            module_logger.propagate = True
            module_logger.handlers = []

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = True


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run model fitting on datasets")

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing datasets",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save results",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["mult_mlp", "mult_vae", "rf"],
        required=True,
        help="Model type to fit",
    )
    parser.add_argument(
        "--worker_node_id",
        type=int,
        required=False,
        default=os.getenv("SLURM_ARRAY_TASK_ID") or 0,
        help="The array ID of the worker node running this process",
    )
    parser.add_argument(
        "--max_worker_nodes",
        type=int,
        required=False,
        default=1,
        help="The total number of worker nodes allocated to this job",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=32,
        help="Maximum number of parallel workers (default: 32 for a typical node)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        nargs="+",
        default=[0.3],
        help="Dropout rate for deep learning models",
    )
    parser.add_argument(
        "--hidden_dim_base",
        type=int,
        nargs="+",
        default=[8],
        help="Smallest hidden dimension size for deep learning models",
    )
    parser.add_argument(
        "--source_epochs",
        type=int,
        nargs="+",
        default=[50],
        help="Number of training epochs for source domain",
    )
    parser.add_argument(
        "--target_epochs",
        type=int,
        nargs="+",
        default=[50],
        help="Number of training epochs for target domain",
    )
    parser.add_argument(
        "--freeze",
        type=str,
        nargs="+",
        default=["none"],
        choices=["none", "marginal"],
        help="Freezing strategy for transfer learning",
    )
    parser.add_argument(
        "--z_dim_base",
        type=int,
        nargs="+",
        default=[8],
        help="Smallest latent dimension size for VAE models",
    )
    parser.add_argument(
        "--n_latent_dims",
        type=int,
        nargs="+",
        default=[2],
        help="Number of latent dimensions",
    )
    parser.add_argument(
        "--init_lr",
        type=float,
        nargs="+",
        default=[0.01],
        help="Number of latent dimensions",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        nargs="+",
        default=[0.01],
        help="Number of latent dimensions",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        nargs="+",
        default=[0.01],
        help="Number of latent dimensions",
    )
    parser.add_argument(
        "--no_tune",
        action="store_true",
        help="Disable hyperparameter tuning (use fixed parameters)",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        help="Path to log file (optional)",
    )
    parser.add_argument(
        "--disable_progress_bar",
        action="store_true",
        help="Disable tqdm progress bar",
    )

    return parser.parse_args()


def setup_param_grid(args: argparse.Namespace) -> dict[str, list[Any]] | None:
    """Set up parameter grid for hyperparameter tuning."""
    if args.no_tune:
        return None
    return {
        "dropout": args.dropout,
        "hidden_dim_base": args.hidden_dim_base,
        "source_epochs": args.source_epochs,
        "target_epochs": args.target_epochs,
        "freeze": args.freeze,
        "z_dim_base": args.z_dim_base,
        "n_latent_dims": args.n_latent_dims,
        "lr": args.init_lr,
        "weight_decay": args.weight_decay,
        "gamma": args.gamma,
    }


def run_model_on_dataset(
    dataset_container: DatasetContainer,
    model_type: str,
    param_grid: dict[str, list[Any]] | None = None,
) -> pd.DataFrame:
    """Run a specific model on a dataset."""
    if model_type == "rf":
        load_r_functions("../..")
        out, _ = fit_rf_model(dataset_container)
        return out
    out, _, _ = fit_dl_model(
        dataset_container,
        model_type,
        device("cpu"),
        param_grid,
    )
    return out


def configure_worker_logging():
    """Configure logging for worker processes."""
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        root.addHandler(handler)
        root.setLevel(logging.INFO)


def process_dataset(args_tuple: tuple) -> None:
    """Process a single dataset with the specified model."""
    configure_worker_logging()

    data_id, data_dir, model_type, param_grid, output_dir, timestamp = args_tuple
    worker_logger = logging.getLogger(f"worker.{data_id[0]}_{data_id[1]}")

    worker_logger.info(f"Starting processing dataset {data_id} with model {model_type}")

    replicate, scenario = data_id

    try:
        dataset_manager = DatasetManager(Path(data_dir))
        dataset_manager.scan_directory()

        dataset_container = dataset_manager.load_dataset_container(data_id)
        dataset_container.set_response_column("response")

        results = run_model_on_dataset(dataset_container, model_type, param_grid)

        output_filename = f"{replicate}_{scenario}_{model_type}.csv"
        output_path = Path(output_dir) / output_filename
        results.to_csv(output_path, index=False)

        worker_logger.info(f"Results saved to {output_path}")
        return data_id
    except Exception as e:
        worker_logger.error(f"Error processing dataset {data_id}: {e!s}", exc_info=True)
        return None


def main() -> None:
    """Run model fitting using multiprocessing pool with tqdm progress bar."""
    if hasattr(mp, "set_start_method"):
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            # Method already set
            pass

    args = parse_args()
    setup_logging(args.log_file)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running with arguments: {args}")

    param_grid = None
    if args.model_type != "rf":
        param_grid = setup_param_grid(args)
        if param_grid:
            logger.info(f"Parameter grid: {param_grid}")
        else:
            logger.info("Using default parameters")
    else:
        from omicsTL.transfer_forest import load_r_functions

        load_r_functions("../..")

    logger.info(f"Detected {os.cpu_count()} available workers.")

    dataset_manager = DatasetManager(data_dir)
    dataset_manager.scan_directory()
    data_ids = dataset_manager.get_available_ids()
    logger.info(f"Found {len(data_ids)} datasets")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    max_workers = args.max_workers
    logger.info(f"Using {max_workers} workers")

    num_parallel_tasks = min(max_workers, len(data_ids))
    logger.info(f"Running {num_parallel_tasks} parallel tasks")

    # Setup parallel worker nodes
    max_worker_nodes = args.max_worker_nodes
    worker_node_id = args.worker_node_id
    if max_worker_nodes > 1:
        logger.info(f"Running as node {worker_node_id} of {max_worker_nodes} worker nodes")

    # Set environment variables to restrict PyTorch threading per process
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    process_args = [
        (data_id, str(data_dir), args.model_type, param_grid, str(output_dir), timestamp) for data_id in data_ids
    ]

    if max_worker_nodes > 1:
        # Subset jobs to only the ones that this worker node should process
        process_args = [
            process_args[i] for i in range(len(process_args)) if i % max_worker_nodes == (worker_node_id - 1)
        ]

        logger.info(f"This node is processing {len(process_args)} datasets")
    with mp.Pool(processes=max_workers, initializer=configure_worker_logging) as pool:
        results = []

        with tqdm(
            total=len(data_ids),
            disable=args.disable_progress_bar,
            desc=f"Processing {args.model_type}",
            unit="dataset",
        ) as pbar:
            for result in pool.imap_unordered(process_dataset, process_args):
                if result is not None:
                    logger.info(f"Completed dataset {result}")
                results.append(result)
                pbar.update()

    successful = sum(1 for r in results if r is not None)
    logger.info(f"Processing complete: {successful} of {len(process_args)} datasets processed successfully")


if __name__ == "__main__":
    main()
