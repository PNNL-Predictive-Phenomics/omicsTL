import random

import pandas as pd
import numpy as np
import torch
from pandas.testing import assert_frame_equal
from torch import device

from omicstl.r_utils import set_seed
from omicstl.simulation_utils.recommendation_utils import recommend_next_batch
from omicstl.simulation_utils.model_utils import fit_dl_model, fit_rf_model
from omicstl.simulation_utils.data_utils import DatasetManager, DatasetContainer
from omicstl.transfer_model import TransferModel

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
        check_names=False,
        rtol=1e-2,
        atol=1e-3,
    )

class ModelTest:
    def run_recommendation(self, model, data: DatasetContainer):
        transfer_model = TransferModel(model)

        existing_data = data.target_data
        response_col = "response"
        feature_cols = [c for c in existing_data.columns if c != response_col]

        feature_ranges = {
            col: (float(existing_data[col].min()), float(existing_data[col].max()))
            for col in feature_cols
        }

        batch = recommend_next_batch(
            model_info=transfer_model,
            existing_data=existing_data,
            response_col=response_col,
            feature_cols=feature_cols,
            feature_ranges=feature_ranges,
            n_candidates=200,
            batch_size=5,
            seed=42,
        )

        # Basic end-to-end sanity checks
        assert type(batch) is pd.DataFrame
        assert not batch.empty
        expected_cols = set(feature_cols) | {
            "predicted_mean", "predicted_std", "acquisition_score", "batch_rank"
        }
        assert expected_cols.issubset(batch.columns)
        assert len(batch) <= 5
        return batch

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

        data = self.load_reg_data(0)

        res, model, _ = fit_dl_model(
            data,
            "mult_vae",
            device("cpu"),
            param_grid,
        )

        assert_results_equal(res, "tests/test_data/test_sample/result/regtest_1_vae.csv")

        self.run_recommendation(model, data)

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

        data = self.load_reg_data(1)

        res, model, _ = fit_dl_model(
            data,
            "mult_vae",
            device("cpu"),
            param_grid,
        )

        assert_results_equal(res, "./tests/test_data/test_sample/result/regtest_2_vae.csv")

        self.run_recommendation(model, data)

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

        data = self.load_reg_data(0)

        res, model, _ = fit_dl_model(
            data,
            "mult_mlp",
            device("cpu"),
            param_grid,
        )

        assert_results_equal(res, "./tests/test_data/test_sample/result/regtest_1_mlp.csv")

        self.run_recommendation(model, data)

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

        data = self.load_reg_data(1)

        res, model, _ = fit_dl_model(
            data,
            "mult_mlp",
            device("cpu"),
            param_grid,
        )

        assert_results_equal(res, "./tests/test_data/test_sample/result/regtest_2_mlp.csv")

        self.run_recommendation(model, data)

    def test_rf_reg(self):
        random.seed(42)
        np.random.seed(42)
        set_seed(42)
        data = self.load_reg_data(0)
        res, model = fit_rf_model(data)
        assert_results_equal(res, "./tests/test_data/test_sample/result/regtest_1_rf.csv")
        self.run_recommendation(model, data)

    def test_rf_cat(self):
        random.seed(42)
        np.random.seed(42)
        set_seed(42)
        data = self.load_reg_data(1)
        res, model = fit_rf_model(data)
        assert_results_equal(res, "./tests/test_data/test_sample/result/regtest_2_rf.csv")
        self.run_recommendation(model, data)

def test_recommend_vae_cat():
    ModelTest().test_vae_cat()

def test_recommend_vae_reg():
    ModelTest().test_vae_reg()

def test_recommend_mlp_cat():
    ModelTest().test_mlp_cat()

def test_recommend_mlp_reg():
    ModelTest().test_mlp_reg()

def test_recommend_rf_cat():
    ModelTest().test_rf_cat()

def test_recommend_rf_reg():
    ModelTest().test_rf_reg()