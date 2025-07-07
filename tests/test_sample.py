import random
import logging

import pandas as pd
import torch
from pandas.testing import assert_frame_equal
from torch import device

from omicsTL.simulation_utils.data_utils import DatasetContainer, DatasetManager
from omicsTL.simulation_utils.model_utils import fit_dl_model, fit_rf_model


logger = logging.getLogger(__name__)


class RegressionModelTest:
    def load_reg_data(self) -> DatasetContainer:
        dataset_manager = DatasetManager("./tests/test_data/regtest/")
        dataset_manager.scan_directory()
        data_ids = dataset_manager.get_available_ids()
        return dataset_manager.load_dataset_container(data_ids[0])

    def test_vae(self):
        param_grid = {
            "dropout": [0.3],
            "hidden_dim_size": [32, 64],
            "source_epochs": [5],
            "target_epochs": [5],
            "freeze": ["none"],
        }
        random.seed(42)
        torch.manual_seed(42)
        res = fit_dl_model(
            self.load_reg_data(),
            "mult_vae",
            device("cpu"),
            param_grid,
        )
        logger.debug(res)
        logger.debug(pd.read_csv("./tests/test_data/vae_regtest_res.csv", index_col=0))
        assert_frame_equal(
            res,
            pd.read_csv("./tests/test_data/vae_regtest_res.csv", index_col=0),
            check_dtype=False,
            check_index_type=False,
            check_column_type=False,
            check_exact=False,
            check_names=False,
        )

    def test_mlp(self):
        param_grid = {
            "dropout": [0.3],
            "hidden_dim_size": [32, 64],
            "source_epochs": [5],
            "target_epochs": [5],
            "freeze": ["none"],
        }
        random.seed(42)
        torch.manual_seed(42)
        res = fit_dl_model(
            self.load_reg_data(),
            "mult_mlp",
            device("cpu"),
            param_grid,
        )
        assert_frame_equal(res, pd.read_csv("./tests/test_data/mlp_regtest_res.csv"))

    def test_rf(self):
        random.seed(42)
        res = fit_rf_model(self.load_reg_data())
        assert_frame_equal(res, pd.read_csv("./tests/test_data/rf_regtest_res.csv"))


def test():
    RegressionModelTest().test_vae()
    # RegressionModelTest().test_mlp()
    # RegressionModelTest().test_rf()
