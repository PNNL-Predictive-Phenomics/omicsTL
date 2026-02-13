import random
import torch
import numpy as np
from torch import device
from omicstl.simulation_utils.data_utils import DatasetManager
from omicstl.simulation_utils.model_utils import fit_dl_model, fit_rf_model
from omicstl.r_utils import set_seed

def regenerate_test_sample_result():
    error_prefix = "DATA-FOR=test_sample.py: ERROR:"

    dataset_manager = DatasetManager("./test_data/test_sample/data")
    dataset_manager.scan_directory()
    dataset_ids = dataset_manager.get_available_ids()

    expected_data_ids_count = 4
    if len(dataset_ids) != expected_data_ids_count:
        print(f"{error_prefix} Incorrect number of data IDs. Expected {expected_data_ids_count}. Got {len(dataset_ids)}.")
        return
    
    # Load data
    regdata_1 = dataset_manager.load_dataset_container(dataset_ids[0])
    regdata_2 = dataset_manager.load_dataset_container(dataset_ids[1])
    catdata_1 = dataset_manager.load_dataset_container(dataset_ids[2])
    catdata_2 = dataset_manager.load_dataset_container(dataset_ids[3])

    # Set response column
    regdata_1.set_response_column("response")
    regdata_2.set_response_column("response")
    catdata_1.set_response_column("response")
    catdata_2.set_response_column("response")

    # Validate test data
    catdata_1_catexpect = 2
    catdata_1_catcount = len(catdata_1.source_data["response"].unique())
    if catdata_1_catcount != catdata_1_catexpect:
        print(f"{error_prefix} Expected {catdata_1_catexpect} categories in catdata_1. Got {catdata_1_catcount}")
        return

    catdata_2_catexpect = 3
    catdata_2_catcount = len(catdata_2.source_data["response"].unique())
    if catdata_2_catcount != catdata_2_catexpect:
        print(f"{error_prefix} Expected {catdata_2_catexpect} categories in catdata_2. Got {catdata_2_catcount}")
        return

    # Generate vae result
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
        regdata_1,
        "mult_vae",
        device("cpu"),
        param_grid,
    )
    res = res.drop("scenario", axis=1)
    res = res.drop("replicate", axis=1)
    res.to_csv('./test_data/test_sample/result/regtest_1_vae.csv', index=False)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    res, _, _ = fit_dl_model(
        regdata_2,
        "mult_vae",
        device("cpu"),
        param_grid,
    )
    res = res.drop("scenario", axis=1)
    res = res.drop("replicate", axis=1)
    res.to_csv('./test_data/test_sample/result/regtest_2_vae.csv', index=False)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    res, _, _ = fit_dl_model(
        catdata_1,
        "mult_vae",
        device("cpu"),
        param_grid,
    )
    res = res.drop("scenario", axis=1)
    res = res.drop("replicate", axis=1)
    res.to_csv('./test_data/test_sample/result/cattest_1_vae.csv', index=False)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    res, _, _ = fit_dl_model(
        catdata_2,
        "mult_vae",
        device("cpu"),
        param_grid,
    )
    res = res.drop("scenario", axis=1)
    res = res.drop("replicate", axis=1)
    res.to_csv('./test_data/test_sample/result/cattest_2_vae.csv', index=False)

    # Generate mlp result
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    res, _, _ = fit_dl_model(
        regdata_1,
        "mult_mlp",
        device("cpu"),
        param_grid,
    )
    res = res.drop("scenario", axis=1)
    res = res.drop("replicate", axis=1)
    res.to_csv('./test_data/test_sample/result/regtest_1_mlp.csv', index=False)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    res, _, _ = fit_dl_model(
        regdata_2,
        "mult_mlp",
        device("cpu"),
        param_grid,
    )
    res = res.drop("scenario", axis=1)
    res = res.drop("replicate", axis=1)
    res.to_csv('./test_data/test_sample/result/regtest_2_mlp.csv', index=False)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    res, _, _ = fit_dl_model(
        catdata_1,
        "mult_mlp",
        device("cpu"),
        param_grid,
    )
    res = res.drop("scenario", axis=1)
    res = res.drop("replicate", axis=1)
    res.to_csv('./test_data/test_sample/result/cattest_1_mlp.csv', index=False)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    res, _, _ = fit_dl_model(
        catdata_2,
        "mult_mlp",
        device("cpu"),
        param_grid,
    )
    res = res.drop("scenario", axis=1)
    res = res.drop("replicate", axis=1)
    res.to_csv('./test_data/test_sample/result/cattest_2_mlp.csv', index=False)

    # Generate rf result
    random.seed(42)
    np.random.seed(42)
    set_seed(42)
    res, _ = fit_rf_model(regdata_1)
    res = res.drop("scenario", axis=1)
    res = res.drop("replicate", axis=1)
    res.to_csv('./test_data/test_sample/result/regtest_1_rf.csv', index=False)

    random.seed(42)
    np.random.seed(42)
    set_seed(42)
    res, _ = fit_rf_model(regdata_2)
    res = res.drop("scenario", axis=1)
    res = res.drop("replicate", axis=1)
    res.to_csv('./test_data/test_sample/result/regtest_2_rf.csv', index=False)

    random.seed(42)
    np.random.seed(42)
    set_seed(42)
    res, _ = fit_rf_model(catdata_1)
    res = res.drop("scenario", axis=1)
    res = res.drop("replicate", axis=1)
    res.to_csv('./test_data/test_sample/result/cattest_1_rf.csv', index=False)

    random.seed(42)
    np.random.seed(42)
    set_seed(42)
    res, _ = fit_rf_model(catdata_1)
    res = res.drop("scenario", axis=1)
    res = res.drop("replicate", axis=1)
    res.to_csv('./test_data/test_sample/result/cattest_2_rf.csv', index=False)

if __name__ == '__main__':
    regenerate_test_sample_result()