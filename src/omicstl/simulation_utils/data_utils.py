"""Utilities for loading and manipulating simulated multiomics data."""

import logging
import pathlib
import re
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Configure logging
logger = logging.getLogger(__name__)


class DatasetContainer:
    """Store and manage source and target datasets.

    Store source and target datasets and provide functionality for
    splitting the source data into training and testing sets. Can
    be used for arbitrary data, but also contains utilities for working
    with data simulated with this package.
    """

    def __init__(
        self,
        source_data: pd.DataFrame,
        target_data: pd.DataFrame,
        target_test_data: list[pd.DataFrame] | None = None,
        id_tuple: tuple[int, int] | None = None,
    ) -> None:
        """Initialize with source and target data.

        Args:
            source_data: DataFrame containing source data
            target_data: DataFrame containing target training data
            response_id: A string indicating the response column in both source, and target datasets.
            target_test_data: Optional list of DataFrames containing target domain test data
            id_tuple: Optional tuple of (first_id, last_id) identifying this dataset. Used for simulated data.

        """
        self.source_data = source_data
        self.target_data = target_data
        self.target_test_data = target_test_data if target_test_data is not None else []
        self.id_tuple = id_tuple if id_tuple else (None, None)
        self.response_id: str

        self.source_train_data: pd.DataFrame | None = None
        self.source_test_data: pd.DataFrame | None = None

    def set_response_column(self, response_id: str) -> None:
        """Set response column for target and testing datasets."""
        self.response_id = response_id


    def split_source_data(
        self,
        test_size: float = 0.2,
        random_state: int | None = None,
        stratify_column: str | None = None
    ) -> None:
        """Split the source data into training and testing sets.

        Args:
            test_size: Proportion of the data to use for testing (default: 0.2)
            random_state: Random seed for reproducibility
            stratify_column: Column name to use for stratified sampling

        Raises:
            ValueError: If stratify_column is specified but not found in source_data

        """
        stratify = None
        if stratify_column is not None:
            if stratify_column not in self.source_data.columns:
                msg = f"Stratify column '{stratify_column}' not found in source data"
                raise ValueError(msg)
            stratify = self.source_data[stratify_column]

        self.source_train_data, self.source_test_data = train_test_split(
            self.source_data,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

    def get_all_data(self) -> dict[str, pd.DataFrame | list[pd.DataFrame]]:
        """Return a dictionary containing all datasets.

        Returns:
            Dictionary containing all datasets:
                - "source": Full source data
                - "source_train": Source training data (if split)
                - "source_test": Source testing data (if split)
                - "target": Target training data
                - "target_test": List of target test datasets

        """
        result: dict[str, pd.DataFrame | list[pd.DataFrame]] = {
            "source": self.source_data,
            "target": self.target_data,
            "target_test": self.target_test_data,
        }

        if self.source_train_data is not None:
            result["source_train"] = self.source_train_data

        if self.source_test_data is not None:
            result["source_test"] = self.source_test_data

        return result

    def is_classification(self) -> bool:
        """Return true if the data is a classification setting."""
        if not hasattr(self, "response_id"):
            msg = "No response_id set, run set_response_column first."
            raise AttributeError(msg)
        return self.source_data[self.response_id].dtype == np.int64

    def __repr__(self) -> str:
        """Return string representation of the DatasetContainer.

        Returns:
            String describing the container and its datasets

        """
        id_info = f" for ID {self.id_tuple}" if self.id_tuple else ""
        split_status = "split" if self.source_train_data is not None else "not split"

        return (
            f"DatasetContainer{id_info} with:\n"
            f"  Source data: {self.source_data.shape} ({split_status})\n"
            f"  Target data: {self.target_data.shape}\n"
            f"  Target test data: {len(self.target_test_data)} datasets"
        )


class DatasetManager:
    """Manage and load simulated datasets organized by replicate/scenario IDs.

    Handle finding, grouping, and loading datasets that follow the naming convention:
    source_data_{first_id}_{last_id}.csv, target_data_{first_id}_{last_id}.csv,
    and target_test_data_{first_id}_{last_id}.csv
    """

    def __init__(self, directory_path: str | pathlib.Path) -> None:
        """Initialize with a directory path.

        Args:
            directory_path: Path to the directory containing the CSV files

        """
        # Store directory path (convert to Path object internally if string is provided)
        self.directory_path = directory_path
        self.grouped_paths: dict[tuple[int, int], dict[str, Any]] | None = None

    def scan_directory(self) -> "DatasetManager":
        """Scan the directory and group CSV files based on their scenario and replicated ids.

        Returns:
            self: For method chaining

        """
        grouped_paths: dict[tuple[int, int], dict[str, str | None | list[str]]] = defaultdict(
            lambda: {"source": None, "target": None, "target_test": []},  # Initialize as empty list
        )
        pattern = r"(source|target|target_test)_data_(\d+)_(\d+)\.csv"
        dir_path = pathlib.Path(self.directory_path)

        for file_path in dir_path.iterdir():
            if not file_path.is_file() or not file_path.name.endswith(".csv"):
                continue
            match = re.match(pattern, file_path.name)
            if not match:
                continue
            file_type, first_num, last_num = match.groups()
            first_num, last_num = int(first_num), int(last_num)
            key = (first_num, last_num)

            # Add to the appropriate category
            if file_type == "source":
                grouped_paths[key]["source"] = str(file_path)
            elif file_type == "target":
                grouped_paths[key]["target"] = str(file_path)
            elif file_type == "target_test":
                grouped_paths[key]["target_test"].append(str(file_path))

        self.grouped_paths = dict(grouped_paths)
        return self

    def get_available_ids(self) -> list[tuple[int, int]]:
        """Return a list of all available ID tuples.

        Returns:
            List of tuples representing available (first_id, last_id) combinations

        Raises:
            RuntimeError: If scan_directory has not been called yet

        """
        if self.grouped_paths is None:
            msg = "Directory has not been scanned yet. Call scan_directory first."
            raise RuntimeError(msg)

        return list(self.grouped_paths.keys())

    def has_complete_dataset(self, id_tuple: tuple[int, int]) -> bool:
        """Check if a complete dataset exists for the given ID tuple.

        Args:
            id_tuple: A tuple of (first_id, last_id) to check

        Returns:
            True if both source and target datasets exist, False otherwise

        Raises:
            RuntimeError: If scan_directory has not been called yet

        """
        if self.grouped_paths is None:
            msg = "Directory has not been scanned yet. Call scan_directory first."
            raise RuntimeError(msg)

        if id_tuple not in self.grouped_paths:
            return False

        paths = self.grouped_paths[id_tuple]
        return paths["source"] is not None and paths["target"] is not None

    def load_datasets(
        self,
        id_tuple: tuple[int, int],
        read_csv_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, pd.DataFrame | list[pd.DataFrame]]:
        """Load datasets corresponding to a specific tuple of replicate/scenario IDs.

        Args:
            id_tuple: A tuple of (first_id, last_id) specifying which datasets to load
            read_csv_kwargs: Additional arguments to pass to pd.read_csv

        Returns:
            A dictionary containing the loaded datasets with keys:
                - 'source': DataFrame for source data
                - 'target': DataFrame for target training data
                - 'target_test': List of DataFrames for target test data

        Raises:
            RuntimeError: If scan_directory has not been called yet
            FileNotFoundError: If any of the required datasets are not found
            ValueError: If the specified id_tuple doesn't exist in the directory

        """
        if self.grouped_paths is None:
            msg = "Directory has not been scanned yet. Call scan_directory first."
            raise RuntimeError(msg)

        if id_tuple not in self.grouped_paths:
            available_ids = self.get_available_ids()
            msg = f"ID tuple {id_tuple} not found. Available ID tuples: {available_ids}"
            raise ValueError(msg)

        paths = self.grouped_paths[id_tuple]

        # Initialize read_csv kwargs if None
        if read_csv_kwargs is None:
            read_csv_kwargs = {}

        # Initialize result dictionary
        datasets: dict[str, pd.DataFrame | list[pd.DataFrame]] = {"source": pd.DataFrame(), "target": pd.DataFrame(), "target_test": []}

        # Load source dataset if available
        if paths["source"]:
            datasets["source"] = pd.read_csv(paths["source"], **read_csv_kwargs)
        else:
            msg = f"Source dataset for ID tuple {id_tuple} not found"
            raise FileNotFoundError(msg)

        # Load target dataset if available
        if paths["target"]:
            datasets["target"] = pd.read_csv(paths["target"], **read_csv_kwargs)
        else:
            msg = f"Target dataset for ID tuple {id_tuple} not found"
            raise FileNotFoundError(msg)

        # Load target test datasets if available
        if paths["target_test"]:
            test_dfs: list[pd.DataFrame] = []
            for test_path in paths["target_test"]:
                test_dfs.append(pd.read_csv(test_path, **read_csv_kwargs))
            datasets["target_test"] = test_dfs
        else:
            logger.warning("No target test datasets found for ID tuple %s", id_tuple)

        return datasets

    def load_dataset_container(
        self,
        id_tuple: tuple[int, int],
        read_csv_kwargs: dict[str, Any] | None = None,
    ) -> DatasetContainer:
        """Load datasets and return them wrapped in a DatasetContainer.

        Args:
            id_tuple: A tuple of (first_id, last_id) specifying which datasets to load
            read_csv_kwargs: Additional arguments to pass to pd.read_csv

        Returns:
            DatasetContainer containing the loaded datasets

        Raises:
            RuntimeError: If scan_directory has not been called yet
            FileNotFoundError: If any of the required datasets are not found
            ValueError: If the specified id_tuple doesn't exist in the directory

        """
        # Load the datasets using the existing method
        datasets = self.load_datasets(id_tuple, read_csv_kwargs)

        source_data=datasets["source"]
        # print(source_data)
        # print(type(source_data))
        if not isinstance(source_data, pd.DataFrame):
            msg = f"Read source data is of type {type(source_data)} instead of pd.DataFrame"
            raise ValueError(msg)

        target_data=datasets["target"]
        if not isinstance(target_data, pd.DataFrame):
            msg = "Read source data is not dataframe."
            raise ValueError(msg)

        target_test_data=datasets["target_test"]
        if not isinstance(target_test_data, list):
            msg = "Read source data is not dataframe."
            raise ValueError(msg)

        # Create and return a DatasetContainer
        return DatasetContainer(
            source_data=source_data,
            target_data=target_data,
            target_test_data=target_test_data,
            id_tuple=id_tuple,
        )

    def load_all_dataset_containers(
        self,
        read_csv_kwargs: dict[str, Any] | None = None,
    ) -> dict[tuple[int, int], DatasetContainer]:
        """Load all available complete datasets into DatasetContainers.

        Args:
            read_csv_kwargs: Additional arguments to pass to pd.read_csv

        Returns:
            Dictionary mapping ID tuples to their respective DatasetContainers

        Raises:
            RuntimeError: If scan_directory has not been called yet

        """
        if self.grouped_paths is None:
            msg = "Directory has not been scanned yet. Call scan_directory first."
            raise RuntimeError(msg)

        containers = {}

        for id_tuple in self.grouped_paths:
            if self.has_complete_dataset(id_tuple):
                try:
                    container = self.load_dataset_container(id_tuple, read_csv_kwargs)
                    containers[id_tuple] = container
                except (FileNotFoundError, ValueError):
                    logger.warning("Skipping ID tuple %s due to error", id_tuple)

        return containers
