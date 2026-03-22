from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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
    
    df["config_name"] = df["config_name"].cat.rename_categories({
        "Int8WeightOnlyConfig": "Int8,<br>Weight Only",
        "Float8WeightOnlyConfig": "Float8,<br>Weight Only",
        "Float8DynamicActivationFloat8WeightConfig": "Float8,<br>Weight and Activation,<br>Dynamic",
        "Float8StaticActivationFloat8WeightConfig": "Float8,<br>Weight and Activation,<br>Static",
        "Int8DynamicActivationInt8WeightConfig": "Int8,<br>Weight and Activation,<br>Dynamic",
        "Float8DynamicActivationInt4WeightConfig": "Int4<br>Weight,<br>Float8 Activation",
        "Int4WeightOnlyConfig": "Int4,<br>Weight Only"
    })

    return df

full_df = convert_str_to_categorical(full_df)
# full branch only processes configs on PyTorch and estimates size
full_df = full_df.drop("runtime", axis=1)

weight_df = convert_str_to_categorical(weight_df)
# only weight-only configs exist in this dataframe
weight_df = weight_df.drop("weight_only", axis=1)

# figure size template
font_size_layout = go.Layout(
    title=dict(font=dict(size=36)),
    legend=dict(font=dict(size=30)),
    xaxis=dict(
        title=dict(font=dict(size=26)),
        tickfont=dict(size=22),
    ),
    yaxis=dict(
        title=dict(font=dict(size=26)),
        tickfont=dict(size=22),
    ),
)

font_size_template = dict(layout=font_size_layout)

# graphs

relative_mae_full_test_only = full_df.query(expr="base_metric == 'MAE' & relative & split == 'test'")
mae_violin = px.violin(data_frame=relative_mae_full_test_only,
                       x="location",
                       y="value",
                       color="device",
                       color_discrete_sequence=px.colors.qualitative.D3,
                       points="all",
                       labels=dict(
                           device="Device",
                           value="Mean Absolute Test Error, Relative to Baseline",
                           location="Location"
                       ),
                       range_y = [0.99, 1.02],
                       box=False,
                       title="No Meaningful Differences In Test Error Exist Between Device and Location")
mae_violin.update_layout(
    template=font_size_template,
    margin=dict(l=120)
)
mae_violin.update_annotations(font=dict(size=26))
mae_violin.write_html(OUTPUT_FIGURE_PATH / "relative_MAE_location_device_test.html")


mae_violin = px.violin(data_frame=relative_mae_full_test_only,
                       x="precision",
                       y="value",
                       color="device",
                       color_discrete_sequence=px.colors.qualitative.D3,
                       points="all",
                       labels=dict(
                           device="Device",
                           value="Mean Absolute Test Error, Relative to Baseline",
                           precision="Data Type"
                       ),
                       range_y = [0.99, 1.02],
                       box=False,
                       title="Both Available Precisions Perform Similarly in Test Error")
mae_violin.update_layout(
    template=font_size_template,
    margin=dict(l=120)
)
mae_violin.update_annotations(font=dict(size=26))
mae_violin.write_html(OUTPUT_FIGURE_PATH / "relative_MAE_precision_device_test.html")

mae_violin = px.violin(data_frame=relative_mae_full_test_only,
                       x="config_name",
                       y="value",
                       color="device",
                       color_discrete_sequence=px.colors.qualitative.D3,
                       points="all",
                       labels=dict(
                           device="Device",
                           value="Mean Absolute Test Error, Relative to Baseline",
                           config_name="PTQ Configuration"
                       ),
                       range_y = [0.99, 1.02],
                       box=False,
                       title="PTQ Approaches Differ With Respect to Test Error")
mae_violin.update_layout(
    template=font_size_template,
    margin=dict(l=120, b=120)
)
mae_violin.update_layout(xaxis_tickfont=dict(size=16))
mae_violin.update_annotations(font=dict(size=26))
mae_violin.write_html(OUTPUT_FIGURE_PATH / "relative_MAE_config_device_test.html")