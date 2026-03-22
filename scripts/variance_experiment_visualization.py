from pathlib import Path

import pandas as pd
import plotly.express as px

# global directory definition
OUTPUT_DATA_PATH = Path.cwd() / "data" / "output" / "csv"
OUTPUT_FIGURE_PATH = Path.cwd() / "data" / "output" / "figures"

# file path definitions
LOCAL_CPU_FULL_RESULT_PATH = (
    OUTPUT_DATA_PATH / "full_ptq_baseline_experiment_data_CPU.csv"
)
LOCAL_GPU_FULL_RESULT_PATH = (
    OUTPUT_DATA_PATH / "full_ptq_baseline_experiment_data_local_GPU.csv"
)
CLUSTER_GPU_FULL_RESULT_PATH = (
    OUTPUT_DATA_PATH / "full_ptq_baseline_experiment_1_data_cluster.csv"
)

LOCAL_CPU_WEIGHT_ONLY_RESULT_PATH = (
    OUTPUT_DATA_PATH / "weight_only_ptq_baseline_experiment_data_CPU.csv"
)
LOCAL_GPU_WEIGHT_ONLY_RESULT_PATH = (
    OUTPUT_DATA_PATH / "weight_only_ptq_baseline_experiment_data_local_GPU.csv"
)
CLUSTER_GPU_WEIGHT_ONLY_RESULT_PATH = (
    OUTPUT_DATA_PATH / "weight_only_ptq_baseline_experiment_1_data_cluster.csv"
)

# importing the dataframes and adding disambiguating variables
local_cpu_full_df = pd.read_csv(LOCAL_CPU_FULL_RESULT_PATH, index_col=0).assign(
    device="CPU", location="Local"
)
local_gpu_full_df = pd.read_csv(LOCAL_CPU_FULL_RESULT_PATH, index_col=0).assign(
    device="GPU", location="Local"
)
cluster_gpu_full_df = pd.read_csv(CLUSTER_GPU_FULL_RESULT_PATH, index_col=0).assign(
    device="GPU", location="Cluster"
)

local_cpu_weight_df = pd.read_csv(
    LOCAL_CPU_WEIGHT_ONLY_RESULT_PATH, index_col=0
).assign(device="CPU", location="Local")
local_gpu_weight_df = pd.read_csv(
    LOCAL_GPU_WEIGHT_ONLY_RESULT_PATH, index_col=0
).assign(device="GPU", location="Local")
cluster_gpu_weight_df = pd.read_csv(
    CLUSTER_GPU_WEIGHT_ONLY_RESULT_PATH, index_col=0
).assign(device="GPU", location="Cluster")

# combining the dataframes
full_df = pd.concat([local_cpu_full_df, local_gpu_full_df, cluster_gpu_full_df])
weight_df = pd.concat([local_cpu_weight_df, local_gpu_weight_df, cluster_gpu_weight_df])

# processing the dataframes

# helper  function for processing
def convert_str_to_categorical(df: pd.DataFrame) -> pd.DataFrame:

    CAT_VAR_LIST = [
        "config_name",
        "precision",
        "metric",
        "base_metric",
        "split",
        "device",
        "location",
        "train_run",
        "eval_run",
    ]
    
    # convert the correct non-Categorical Series to Categorical
    for variable in CAT_VAR_LIST:
        df[variable] = df[variable].astype("category")
        
    # explicitly converting the boolean Series into a Categorical
    df["calibration"] = (df["dynamic_calibration"]
                         .map({True: "dynamic", False: "static"})
                         .astype("category"))
    
    # dropping redundant or irrelevant columns
    df = df.drop(labels=["dynamic_calibration",
                         "metric",
                         "bits_per_weight"], axis=1)

    return df

full_df = convert_str_to_categorical(full_df)
# full branch only processes configs on PyTorch and estimates size
full_df = full_df.drop("runtime", axis=1)

weight_df = convert_str_to_categorical(weight_df)
# only weight-only configs exist in this dataframe
weight_df = weight_df.drop("weight_only", axis=1)

# graphs
relative_mae_full_df = full_df.query(expr="base_metric == 'MAE' & relative")
mae_violin = px.violin(data_frame=relative_mae_full_df,
                       x="device",
                       y="value",
                       color="location")
