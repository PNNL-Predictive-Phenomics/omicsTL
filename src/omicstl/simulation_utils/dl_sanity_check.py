"""Docstring."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import omicsTL.simulation_utils.model_utils as mu
from omicsTL.simulation_utils.data_utils import DatasetManager


def plot_df_columns_over_time(df, figsize=(12, 6), use_subplots=False, skip_non_numeric=True):
    if skip_non_numeric:
        columns_to_plot = df.select_dtypes(include=[np.number]).columns
    else:
        columns_to_plot = df.columns

    time_axis = np.arange(len(df))

    if use_subplots and len(columns_to_plot) > 1:
        fig, axes = plt.subplots(len(columns_to_plot), 1, figsize=figsize, sharex=True)

        for i, column in enumerate(columns_to_plot):
            ax = axes[i] if len(columns_to_plot) > 1 else axes
            ax.plot(time_axis, df[column], label=column)
            ax.set_ylabel("Value")
            ax.set_title(column)
            ax.grid(True)

        (axes[-1] if len(columns_to_plot) > 1 else axes).set_xlabel("Time (Row Number)")

    else:
        fig, ax = plt.subplots(figsize=figsize)

        for column in columns_to_plot:
            ax.plot(time_axis, df[column], label=column)

        ax.set_xlabel("Time (Row Number)")
        ax.set_ylabel("Value")
        ax.set_title("Column Values Over Time")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()

    return fig


dataset_manager = DatasetManager("./data/simulated_data/simpletest2/")
dataset_manager.scan_directory()
data_ids = dataset_manager.get_available_ids()
dataset_container = dataset_manager.load_dataset_container(data_ids[20])

results = mu.create_results_df()
replicate, scenario = dataset_container.id_tuple
is_classification = dataset_container.is_classification()
feature_cols = mu.get_feature_columns(dataset_container.source_data)

source_train_samples, source_validation_samples = mu.split_dataframe_indices(dataset_container.source_data, 0.8)

source_train_partition = mu.create_data_partition(
    data=dataset_container.source_data,
    feature_cols=feature_cols,
    row_ids=source_train_samples,
)

source_validation_partition = mu.create_data_partition(
    data=dataset_container.source_data,
    feature_cols=feature_cols,
    row_ids=source_validation_samples,
)

target_train_samples, target_validation_samples = mu.split_dataframe_indices(dataset_container.target_data, 0.8)

target_full = mu.create_data_partition(
    data=dataset_container.target_data,
    feature_cols=feature_cols,
)

target_train_partition = mu.create_data_partition(
    data=dataset_container.target_data,
    feature_cols=feature_cols,
    row_ids=target_train_samples,
)

target_validation_partition = mu.create_data_partition(
    data=dataset_container.target_data,
    feature_cols=feature_cols,
    row_ids=target_validation_samples,
)

source_partition = {"training": source_train_partition, "early_stopping": source_validation_partition}
target_partition = {"training": target_full, "early_stopping": None}
target_partition_nosource = {"training": target_train_partition, "early_stopping": target_validation_partition}

hyperparams = {
    "dropout": 0.2,
    "hidden_dim_base": 12,
    "z_dim_base": 12,
    "n_latent_dims": 2,
    "source_epochs": 500,
    "target_epochs": 500,
    "freeze": "none",
    "lr": 0.01,
}

model_type = "mult_vae"
output_dim = dataset_container.source_data["response"].nunique() if is_classification else 1
torch_device = torch.device("cpu")
config = mu.ModelConfig(
    model_type=model_type,
    hyperparams=hyperparams,
    is_classification=is_classification,
    torch_device=torch_device,
    output_dim=output_dim,
)

model = mu.create_model(config)

lr = config.hyperparams.get("lr", 0.001)
mu.train_model(model=model, data=source_partition, config=config, model_id="source", lr=lr)
mu.train_model(model=model, data=target_partition, config=config, model_id="target", source_model="source", lr=lr)

config_nosource = mu.ModelConfig(
    model_type=model_type,
    hyperparams=hyperparams,
    is_classification=is_classification,
    torch_device=torch_device,
    output_dim=output_dim,
)
model_nosource = mu.create_model(config_nosource)
mu.train_model(
    model=model_nosource,
    data=target_partition_nosource,
    config=config_nosource,
    model_id="target_nosource",
    lr=lr,
)


# make test results, plot trajectory

for i, test_dataset in enumerate(dataset_container.target_test_data):
    test_data = mu.create_data_partition(test_dataset, feature_cols)

    test_context = mu.EvaluationContext(
        scenario=scenario,
        replicate=replicate,
        split_name=f"test_{i}",
        model_type=model_type,
        hyperparams=hyperparams,
    )
    test_results = mu.evaluate_model(model=model, data=test_data, model_id="target", context=test_context)
    results = pd.concat([results, test_results], ignore_index=True)

    test_context = mu.EvaluationContext(
        scenario=scenario,
        replicate=replicate,
        split_name=f"test_{i}",
        model_type=model_type,
        hyperparams=hyperparams,
    )
    test_results = mu.evaluate_model(
        model=model_nosource,
        data=test_data,
        model_id="target_nosource",
        context=test_context,
    )
    results = pd.concat([results, test_results], ignore_index=True)

results


max_length = max(
    len(model.source.loss_history),
    len(model.source.train_metric),
    len(model.target.loss_history),
    len(model.target.train_metric),
    len(model_nosource.target_nosource.loss_history),
    len(model_nosource.target_nosource.train_metric),
)

idx = pd.RangeIndex(max_length)
df = pd.DataFrame(
    {
        "source_loss": pd.Series(model.source.loss_history).reindex(idx).ffill(),
        "target_loss": pd.Series(model.target.loss_history).reindex(idx).ffill(),
        "nosource_loss": pd.Series(model_nosource.target_nosource.loss_history).reindex(idx).ffill(),
    },
)
plot_df_columns_over_time(
    df,
)

idx = pd.RangeIndex(max_length)
df = pd.DataFrame(
    {
        "source_metric": pd.Series(model.source.train_metric).reindex(idx).ffill(),
        "target_metric": pd.Series(model.target.train_metric).reindex(idx).ffill(),
        "nosource_metric": pd.Series(model_nosource.target_nosource.train_metric).reindex(idx).ffill(),
    },
)
plot_df_columns_over_time(
    df,
)
