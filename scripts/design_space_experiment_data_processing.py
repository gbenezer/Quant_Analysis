from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# global directory definition
OUTPUT_DATA_PATH = Path.cwd() / "data" / "output" / "csv"

# paths
full_result_paths = [
    (OUTPUT_DATA_PATH / f"full_ptq_config_sampling_experiment_{i}_data_Cluster.csv")
    for i in range(2, 12)
]

weight_only_paths = [
    (
        OUTPUT_DATA_PATH
        / f"weight_only_ptq_config_sampling_experiment_{i}_data_Cluster.csv"
    )
    for i in range(2, 12)
]

# result df lists
full_result_dfs = [
    pd.read_csv(result_path, index_col=0)
    .dropna()
    .assign(experiment_number=(experiment + 1))
    for experiment, result_path in enumerate(full_result_paths)
]
weight_result_dfs = [
    pd.read_csv(result_path, index_col=0)
    .dropna()
    .assign(experiment_number=(experiment + 1))
    for experiment, result_path in enumerate(weight_only_paths)
]

# result dfs
full_result_df = pd.concat(full_result_dfs)
full_result_df["eval_run"] = full_result_df["eval_run"].astype("int")
full_result_df["model_ID"] = (
    full_result_df["model_ID"] + "_" + full_result_df["experiment_number"].astype(str)
)
weight_only_df = pd.concat(weight_result_dfs)
weight_only_df["eval_run"] = weight_only_df["eval_run"].astype("int")
weight_only_df["model_ID"] = (
    weight_only_df["model_ID"] + "_" + weight_only_df["experiment_number"].astype(str)
)

# saving the files
full_result_df.to_csv((OUTPUT_DATA_PATH / "full_ptq_config_sampling_experiments.csv"))
weight_only_df.to_csv(
    (OUTPUT_DATA_PATH / "weight_only_ptq_config_sampling_experiments.csv")
)
