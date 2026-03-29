from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# global directory definition
OUTPUT_DATA_PATH = Path.cwd() / "data" / "output" / "csv"
OUTPUT_FIGURE_PATH = Path.cwd() / "data" / "output" / "figures"

# file path definition
FULL_RESULT_PATH = (
    OUTPUT_DATA_PATH / "full_ptq_config_sampling_experiment_data_Cluster.csv"
)
WEIGHT_ONLY_RESULT_PATH = (
    OUTPUT_DATA_PATH / "weight_only_ptq_config_sampling_experiment_data_Cluster.csv"
)

# importing the dataframes and adding disambiguating variables
full_df = pd.read_csv(FULL_RESULT_PATH, index_col=0)
weight_df = pd.read_csv(WEIGHT_ONLY_RESULT_PATH, index_col=0)

# processing the dataframes
# helper  function for processing
def convert_str_to_categorical(df: pd.DataFrame) -> pd.DataFrame:

    CAT_VAR_LIST = [
        "config_name",
        "precision",
        "metric",
        "base_metric",
        "eval_run",
        "runtime",
        "activation",
        "model_ID",
    ]

    # convert the correct non-Categorical Series to Categorical
    for variable in CAT_VAR_LIST:
        df[variable] = df[variable].astype("category")

    # explicitly converting the boolean Series into a Categorical
    df["calibration"] = (
        df["dynamic_calibration"]
        .map({True: "dynamic", False: "static"})
        .astype("category")
    )

    # dropping redundant or irrelevant columns
    df = df.drop(labels=["dynamic_calibration", "metric", "bits_per_weight"], axis=1)

    df["config_name"] = df["config_name"].cat.rename_categories(
        {
            "Int8WeightOnlyConfig": "Int8,<br>Weight Only",
            "Float8WeightOnlyConfig": "Float8,<br>Weight Only",
            "Float8DynamicActivationFloat8WeightConfig": "Float8,<br>Weight and Activation,<br>Dynamic",
            "Float8StaticActivationFloat8WeightConfig": "Float8,<br>Weight and Activation,<br>Static",
            "Int8DynamicActivationInt8WeightConfig": "Int8,<br>Weight and Activation,<br>Dynamic",
            "Float8DynamicActivationInt4WeightConfig": "Int4<br>Weight,<br>Float8 Activation",
            "Int4WeightOnlyConfig": "Int4,<br>Weight Only",
        }
    )

    df["runtime"] = df["runtime"].cat.rename_categories(
        {
            "pytorch": "PyTorch",
            "onnx": "ONNX",
            "pt2": "PT2",
        }
    )

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

print(full_df.info())
print(full_df["relative"])
print(weight_df.info())

# # graphs
# relative_mae_full_test_only = full_df.query(
#     expr="base_metric == 'MAE' & relative & split == 'test'"
# )
# mae_violin = px.violin(
#     data_frame=relative_mae_full_test_only,
#     x="location",
#     y="value",
#     color="device",
#     color_discrete_sequence=px.colors.qualitative.D3,
#     points="all",
#     labels=dict(
#         device="Device",
#         value="Mean Absolute Test Error, Relative to Baseline",
#         location="Location",
#     ),
#     range_y=[0.99, 1.02],
#     box=True,
#     title="Distributional Differences In Test Error Between Device and Location",
# )
# mae_violin.update_layout(template=font_size_template, margin=dict(l=120))
# mae_violin.update_annotations(font=dict(size=26))
# mae_violin.write_html(OUTPUT_FIGURE_PATH / "relative_MAE_location_device_test.html")


# mae_violin = px.violin(
#     data_frame=relative_mae_full_test_only,
#     x="precision",
#     y="value",
#     color="device",
#     color_discrete_sequence=px.colors.qualitative.D3,
#     points="all",
#     labels=dict(
#         device="Device",
#         value="Mean Absolute Test Error, Relative to Baseline",
#         precision="Data Type",
#     ),
#     range_y=[0.99, 1.02],
#     box=False,
#     title="Both Available Precisions Perform Similarly in Test Error",
# )
# mae_violin.update_layout(template=font_size_template, margin=dict(l=120))
# mae_violin.update_annotations(font=dict(size=26))
# mae_violin.write_html(OUTPUT_FIGURE_PATH / "relative_MAE_precision_device_test.html")

# mae_violin = px.violin(
#     data_frame=relative_mae_full_test_only,
#     x="config_name",
#     y="value",
#     color="device",
#     color_discrete_sequence=px.colors.qualitative.D3,
#     points="all",
#     labels=dict(
#         device="Device",
#         value="Mean Absolute Test Error, Relative to Baseline",
#         config_name="PTQ Configuration",
#     ),
#     range_y=[0.99, 1.02],
#     box=False,
#     title="PTQ Approaches Differ With Respect to Test Error",
# )
# mae_violin.update_layout(template=font_size_template, margin=dict(l=120, b=120))
# mae_violin.update_layout(xaxis_tickfont=dict(size=16))
# mae_violin.update_annotations(font=dict(size=26))
# mae_violin.write_html(OUTPUT_FIGURE_PATH / "relative_MAE_config_device_test.html")

# mae_violin = px.violin(
#     data_frame=relative_mae_full_test_only,
#     x="config_name",
#     y="value",
#     color="device",
#     color_discrete_sequence=px.colors.qualitative.D3,
#     points="all",
#     facet_col="location",
#     labels=dict(
#         device="Device",
#         value="Mean Absolute Test Error, Relative to Baseline",
#         config_name="PTQ Configuration",
#         location="Location",
#     ),
#     range_y=[0.99, 1.02],
#     box=False,
#     title="PTQ Approaches Differ With Respect to Test Error",
# )
# mae_violin.update_layout(template=font_size_template, margin=dict(l=120, b=120))
# mae_violin.update_layout(xaxis_tickfont=dict(size=16))
# mae_violin.update_annotations(font=dict(size=26))
# mae_violin.write_html(
#     OUTPUT_FIGURE_PATH / "relative_MAE_config_device_location_test.html"
# )

# mae_violin = px.violin(
#     data_frame=relative_mae_full_test_only,
#     x="config_name",
#     y="value",
#     color="location",
#     color_discrete_sequence=px.colors.qualitative.D3,
#     points="all",
#     labels=dict(
#         value="Mean Absolute Test Error, Relative to Baseline",
#         config_name="PTQ Configuration",
#         location="Location",
#     ),
#     range_y=[0.99, 1.02],
#     box=False,
#     title="PTQ Approaches Differ With Respect to Test Error",
# )
# mae_violin.update_layout(template=font_size_template, margin=dict(l=120, b=120))
# mae_violin.update_layout(xaxis_tickfont=dict(size=16))
# mae_violin.update_annotations(font=dict(size=26))
# mae_violin.write_html(OUTPUT_FIGURE_PATH / "relative_MAE_config_location_test.html")

# # absolute median latency
# absolute_median_latency_full = full_df.query(
#     expr="base_metric == 'median_latency' & ~relative"
# )
# absolute_median_latency_full["value"] = absolute_median_latency_full["value"] * 1000.0
# absolute_latency_violin = px.violin(
#     data_frame=absolute_median_latency_full,
#     x="location",
#     y="value",
#     color="device",
#     color_discrete_sequence=px.colors.qualitative.D3,
#     points="all",
#     labels=dict(
#         device="Device",
#         value="Absolute Median Latency, Milliseconds",
#         location="Location",
#     ),
#     box=False,
#     title="No Meaningful Differences In Median Latency Between Device and Location",
# )
# absolute_latency_violin.update_layout(template=font_size_template, margin=dict(l=120))
# absolute_latency_violin.update_annotations(font=dict(size=26))
# absolute_latency_violin.write_html(
#     OUTPUT_FIGURE_PATH / "absolute_median_latency_location_device.html"
# )

# absolute_latency_violin = px.violin(
#     data_frame=absolute_median_latency_full,
#     x="config_name",
#     y="value",
#     color="device",
#     color_discrete_sequence=px.colors.qualitative.D3,
#     points="all",
#     labels=dict(
#         device="Device",
#         value="Absolute Median Latency, Milliseconds",
#         config_name="PTQ Configuration",
#     ),
#     box=False,
#     title="Absolute Median PyTorch Inference Latency Variation Between Configurations",
# )
# absolute_latency_violin.update_layout(
#     template=font_size_template, margin=dict(l=120, b=120)
# )
# absolute_latency_violin.update_layout(xaxis_tickfont=dict(size=16))
# absolute_latency_violin.update_annotations(font=dict(size=26))
# absolute_latency_violin.write_html(
#     OUTPUT_FIGURE_PATH / "absolute_median_latency_config_device.html"
# )

# absolute_latency_violin = px.violin(
#     data_frame=absolute_median_latency_full,
#     x="config_name",
#     y="value",
#     color="location",
#     color_discrete_sequence=px.colors.qualitative.D3,
#     points="all",
#     labels=dict(
#         location="Location",
#         value="Absolute Median Latency, Milliseconds",
#         config_name="PTQ Configuration",
#     ),
#     box=False,
#     title="Absolute Median PyTorch Inference Latency Variation Between Configurations",
# )
# absolute_latency_violin.update_layout(
#     template=font_size_template, margin=dict(l=120, b=120)
# )
# absolute_latency_violin.update_layout(xaxis_tickfont=dict(size=16))
# absolute_latency_violin.update_annotations(font=dict(size=26))
# absolute_latency_violin.write_html(
#     OUTPUT_FIGURE_PATH / "absolute_median_latency_config_location.html"
# )

# relative_latency_violin = px.violin(
#     data_frame=full_df.query(expr="base_metric == 'median_latency' & relative"),
#     x="config_name",
#     y="value",
#     color="device",
#     color_discrete_sequence=px.colors.qualitative.D3,
#     points="all",
#     labels=dict(
#         device="Device",
#         value="Relative Median Latency",
#         config_name="PTQ Configuration",
#     ),
#     box=False,
#     title="Relative Median PyTorch Inference Latency Variation Between Configurations",
# )
# relative_latency_violin.update_layout(
#     template=font_size_template, margin=dict(l=120, b=120)
# )
# relative_latency_violin.update_layout(xaxis_tickfont=dict(size=16))
# relative_latency_violin.update_annotations(font=dict(size=26))
# relative_latency_violin.write_html(
#     OUTPUT_FIGURE_PATH / "relative_median_latency_config_device.html"
# )

# relative_latency_violin = px.violin(
#     data_frame=full_df.query(expr="base_metric == 'median_latency' & relative"),
#     x="config_name",
#     y="value",
#     color="location",
#     color_discrete_sequence=px.colors.qualitative.D3,
#     points="all",
#     labels=dict(
#         location="Location",
#         value="Relative Median Latency",
#         config_name="PTQ Configuration",
#     ),
#     box=False,
#     title="Relative Median PyTorch Inference Latency Variation Between Configurations",
# )
# relative_latency_violin.update_layout(
#     template=font_size_template, margin=dict(l=120, b=120)
# )
# relative_latency_violin.update_layout(xaxis_tickfont=dict(size=16))
# relative_latency_violin.update_annotations(font=dict(size=26))
# relative_latency_violin.write_html(
#     OUTPUT_FIGURE_PATH / "relative_median_latency_config_location.html"
# )

# relative_latency_violin_weight_only = px.violin(
#     data_frame=weight_df.query(expr="base_metric == 'median_latency' & relative"),
#     x="runtime",
#     y="value",
#     color="precision",
#     color_discrete_sequence=px.colors.qualitative.D3,
#     points="all",
#     labels=dict(
#         precision="Data Type", value="Relative Median Latency", runtime="Runtime"
#     ),
#     box=False,
#     title="Relative Median Inference Latency Variation Between Runtimes",
# )
# relative_latency_violin_weight_only.update_layout(
#     template=font_size_template, margin=dict(l=120)
# )
# relative_latency_violin_weight_only.update_annotations(font=dict(size=26))
# relative_latency_violin_weight_only.write_html(
#     OUTPUT_FIGURE_PATH / "relative_median_latency_precision_runtime.html"
# )

# relative_latency_violin_weight_only = px.violin(
#     data_frame=weight_df.query(expr="base_metric == 'median_latency' & relative"),
#     x="runtime",
#     y="value",
#     color="precision",
#     color_discrete_sequence=px.colors.qualitative.D3,
#     points="all",
#     facet_col="location",
#     facet_row="device",
#     labels=dict(
#         precision="Data Type", value="Relative Median Latency", runtime="Runtime"
#     ),
#     box=False,
#     title="Relative Median Inference Latency Variation Between Runtimes",
# )
# relative_latency_violin_weight_only.update_layout(
#     template=font_size_template, margin=dict(l=120)
# )
# relative_latency_violin_weight_only.update_annotations(font=dict(size=26))
# relative_latency_violin_weight_only.write_html(
#     OUTPUT_FIGURE_PATH
#     / "relative_median_latency_precision_runtime_device_location.html"
# )

# relative_size_violin = px.violin(
#     data_frame=weight_df.query(expr="base_metric == 'model_size' & relative"),
#     x="runtime",
#     y="value",
#     color="precision",
#     color_discrete_sequence=px.colors.qualitative.D3,
#     points="all",
#     labels=dict(precision="Data Type", value="Relative File Size", runtime="Runtime"),
#     box=False,
#     title="Relative Size Variation Between Runtimes",
# )
# relative_size_violin.update_layout(template=font_size_template, margin=dict(l=120))
# relative_size_violin.update_annotations(font=dict(size=26))
# relative_size_violin.write_html(
#     OUTPUT_FIGURE_PATH / "relative_size_precision_runtime.html"
# )
# relative_size_df = weight_df.query(expr="base_metric == 'model_size' & relative")
# relative_size_summary_df = relative_size_df.groupby(["runtime", "precision"]).agg(
#     func={"value": [np.median, np.std]}
# )
# print(relative_size_summary_df)
