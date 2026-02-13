import random

import pandas as pd
import numpy as np
import torch
from pandas.testing import assert_frame_equal
from torch import device

from omicstl.simulation_utils.data_utils import DatasetContainer, DatasetManager
from omicstl.simulation_utils.model_utils import fit_dl_model, fit_rf_model
from omicstl.r_utils import set_seed

def assert_results_equal(res: pd.DataFrame, expect_res_path: str):
    compare_res = pd.read_csv(expect_res_path)
    
    res = res.drop("scenario", axis=1)
    res = res.drop("replicate", axis=1)

    assert_frame_equal(
        res,
        compare_res,
        check_dtype=False,
        check_index_type=False,
        check_column_type=False,
        check_exact=False,
        check_names=False
    )

class ModelTest:
    def load_reg_data(self, index) -> DatasetContainer:
        dataset_manager = DatasetManager("./tests/test_data/test_sample/data")
        dataset_manager.scan_directory()
        data_ids = dataset_manager.get_available_ids()

        assert index < 2

        data = dataset_manager.load_dataset_container(data_ids[index])
        data.set_response_column("response")
        return data

    def test_vae_reg(self):
        param_grid = {
            "dropout": [0.3],
            "hidden_dim_size": [32, 64],
            "source_epochs": [5],
            "target_epochs": [5],
            "freeze": ["none"],
        }

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        res, _, _ = fit_dl_model(
            self.load_reg_data(0),
            "mult_vae",
            device("cpu"),
            param_grid,
        )

        assert_results_equal(res, "tests/test_data/test_sample/result/regtest_1_vae.csv")

    def test_vae_cat(self):
        param_grid = {
            "dropout": [0.3],
            "hidden_dim_size": [32, 64],
            "source_epochs": [5],
            "target_epochs": [5],
            "freeze": ["none"],
        }

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        res, _, _ = fit_dl_model(
            self.load_reg_data(1),
            "mult_vae",
            device("cpu"),
            param_grid,
        )

        assert_results_equal(res, "./tests/test_data/test_sample/result/regtest_2_vae.csv")

    def test_mlp_reg(self):
        param_grid = {
            "dropout": [0.3],
            "hidden_dim_size": [32, 64],
            "source_epochs": [5],
            "target_epochs": [5],
            "freeze": ["none"],
        }

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        res, _, _ = fit_dl_model(
            self.load_reg_data(0),
            "mult_mlp",
            device("cpu"),
            param_grid,
        )

        assert_results_equal(res, "./tests/test_data/test_sample/result/regtest_1_mlp.csv")

    def test_mlp_cat(self):
        param_grid = {
            "dropout": [0.3],
            "hidden_dim_size": [32, 64],
            "source_epochs": [5],
            "target_epochs": [5],
            "freeze": ["none"],
        }

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        res, _, _ = fit_dl_model(
            self.load_reg_data(1),
            "mult_mlp",
            device("cpu"),
            param_grid,
        )

        assert_results_equal(res, "./tests/test_data/test_sample/result/regtest_2_mlp.csv")

    def test_rf_reg(self):
        random.seed(42)
        np.random.seed(42)
        set_seed(42)
        res, _ = fit_rf_model(self.load_reg_data(0))
        assert_results_equal(res, "./tests/test_data/test_sample/result/regtest_1_rf.csv")

    def test_rf_cat(self):
        random.seed(42)
        np.random.seed(42)
        set_seed(42)
        res, _ = fit_rf_model(self.load_reg_data(1))
        assert_results_equal(res, "./tests/test_data/test_sample/result/regtest_2_rf.csv")

def test_vae_reg():
    ModelTest().test_vae_reg()
    
def test_mlp_reg():
    ModelTest().test_mlp_reg()
    
def test_rf_reg():
    ModelTest().test_rf_reg()

def test_vae_cat():
    ModelTest().test_vae_cat()

def test_mlp_cat():
    ModelTest().test_mlp_cat()

def test_rf_cat():
    ModelTest().test_rf_cat()
