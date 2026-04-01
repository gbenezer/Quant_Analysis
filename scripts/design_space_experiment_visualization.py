from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# global directory definition
OUTPUT_DATA_PATH = Path.cwd() / "data" / "output" / "csv"
OUTPUT_FIGURE_PATH = Path.cwd() / "data" / "output" / "figures"

# file path definition
FULL_RESULT_PATH = OUTPUT_DATA_PATH / "full_ptq_config_sampling_experiments.csv"
WEIGHT_ONLY_RESULT_PATH = (
    OUTPUT_DATA_PATH / "weight_only_ptq_config_sampling_experiments.csv"
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
        "experiment_number",
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

    df["activation"] = df["activation"].cat.rename_categories(
        {
            "relu": "ReLU",
            "leaky_relu": "Leaky ReLU",
            "gelu": "GELU",
            "elu": "ELU",
            "celu": "CELU",
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

relative_mae_full_data = full_df.query(expr="base_metric == 'MAE' & relative")
relative_mae_violin = px.violin(
    data_frame=relative_mae_full_data,
    x="experiment_number",
    y="value",
    points="all",
    labels=dict(
        value="Mean Absolute Test Error, Relative to Baseline",
        experiment_number="Experiment Number",
    ),
    box=True,
    title="Distributional Differences In Test Error Across Experiments",
)
relative_mae_violin.update_layout(template=font_size_template, margin=dict(l=120))
relative_mae_violin.update_annotations(font=dict(size=26))
relative_mae_violin.write_html(
    OUTPUT_FIGURE_PATH / "relative_MAE_experiment_number.html"
)

number_neurons_violin = px.violin(
    data_frame=relative_mae_full_data,
    x="experiment_number",
    y="total_hidden_neurons",
    points="all",
    labels=dict(
        total_hidden_neurons="Total Number of Hidden Neurons",
        experiment_number="Experiment Number",
    ),
    box=True,
    title="Distributional Differences In Hidden Layer Neurons<br>Across Experiments",
)
number_neurons_violin.update_layout(template=font_size_template, margin=dict(l=120))
number_neurons_violin.update_annotations(font=dict(size=26))
number_neurons_violin.write_html(
    OUTPUT_FIGURE_PATH / "number_neurons_experiment_number.html"
)

number_neurons_histogram = px.histogram(
    data_frame=relative_mae_full_data,
    x="total_hidden_neurons",
    labels=dict(total_hidden_neurons="Total Number of Hidden Neurons"),
    title="Histogram of Hidden Layer Neurons",
    marginal="rug",
    nbins=50,
)
number_neurons_histogram.update_layout(template=font_size_template, margin=dict(l=120))
number_neurons_histogram.write_html(
    OUTPUT_FIGURE_PATH / "number_neurons_histogram.html"
)

neuron_activation_violin = px.violin(
    data_frame=relative_mae_full_data,
    x="activation",
    y="total_hidden_neurons",
    points="all",
    labels=dict(
        total_hidden_neurons="Total Number of Hidden Neurons",
        activation="Activation Function",
    ),
    box=True,
    title="Distributional Differences In Hidden Layer Neurons<br>Across Activation Functions",
)
neuron_activation_violin.update_layout(template=font_size_template, margin=dict(l=120))
neuron_activation_violin.update_annotations(font=dict(size=26))
neuron_activation_violin.write_html(
    OUTPUT_FIGURE_PATH / "number_neurons_activation_violin.html"
)

neuron_per_layer_activation_scatter = px.scatter_3d(
    data_frame=relative_mae_full_data,
    x="hidden_layer_1_neurons",
    y="hidden_layer_2_neurons",
    z="hidden_layer_3_neurons",
    color="activation",
    labels=dict(
        hidden_layer_1_neurons="Neurons, Layer 1",
        hidden_layer_2_neurons="Neurons, Layer 2",
        hidden_layer_3_neurons="Neurons, Layer 3",
        activation="Activation<br>Function",
    ),
    opacity=0.5,
)
neuron_per_layer_activation_scatter.update_layout(
    template=font_size_template,
    margin=dict(l=120),
)
neuron_per_layer_activation_scatter.update_annotations(font=dict(size=26))
neuron_per_layer_activation_scatter.write_html(
    OUTPUT_FIGURE_PATH / "number_neurons_per_layer_activation_scatter_3d.html"
)

fig = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=(
        "Layer 1 vs. Layer 2",
        "Layer 1 vs. Layer 3",
        "Layer 2 vs. Layer 3",
    ),
    column_widths=[800] * 3,
    row_heights=[800],
)
activation_category = relative_mae_full_data["activation"].astype("category")
activation_labels = activation_category.cat.categories.tolist()
color_map = {
    label: px.colors.qualitative.Plotly[i] for i, label in enumerate(activation_labels)
}
colors = activation_category.map(color_map)
fig.add_trace(
    go.Scatter(
        x=relative_mae_full_data["hidden_layer_1_neurons"],
        y=relative_mae_full_data["hidden_layer_2_neurons"],
        marker=dict(color=colors),
        mode="markers",
        showlegend=False,
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=relative_mae_full_data["hidden_layer_1_neurons"],
        y=relative_mae_full_data["hidden_layer_3_neurons"],
        marker=dict(color=colors),
        mode="markers",
        showlegend=False,
    ),
    row=1,
    col=2,
)
fig.add_trace(
    go.Scatter(
        x=relative_mae_full_data["hidden_layer_2_neurons"],
        y=relative_mae_full_data["hidden_layer_3_neurons"],
        marker=dict(color=colors),
        mode="markers",
        showlegend=False,
    ),
    row=1,
    col=3,
)
for label, color in color_map.items():
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(color=color, size=10),
            name=label,
            showlegend=True,
        )
    )
fig.update_layout(
    template=font_size_template,
)
fig.update_layout(
    width=1400,
    height=600,
    margin=dict(l=80, r=80, t=160, b=80),
    title=dict(
        text="Neuron Per Hidden Layer - Activation Function Distribution is <br>Pseudorandom and Low-Discrepancy",
        y=0.97,
        yanchor="top",
        x=0.5,
        xanchor="center",
        font=dict(size=26),  # reduce title font size to avoid crowding
    ),
    legend=dict(font=dict(size=18)),
)
fig.update_xaxes(title_text="Layer 1 Neurons", row=1, col=1)
fig.update_xaxes(title_text="Layer 1 Neurons", row=1, col=2)
fig.update_xaxes(title_text="Layer 2 Neurons", row=1, col=3)

fig.update_yaxes(title_text="Layer 2 Neurons", row=1, col=1)
fig.update_yaxes(title_text="Layer 3 Neurons", row=1, col=2)
fig.update_yaxes(title_text="Layer 3 Neurons", row=1, col=3)
fig.update_annotations(font=dict(size=26))
fig.write_html(
    OUTPUT_FIGURE_PATH / "number_neurons_per_layer_activation_scatter_2d.html"
)


relative_mae_weight_only = weight_df.query(
    expr="base_metric == 'MAE' & relative"
).rename({"value": "relative_MAE"}, axis=1)
relative_size_weight_only_pt2 = weight_df.query(
    expr="base_metric == 'model_size' & relative & runtime == 'PT2'"
).rename({"value": "relative_model_pt2_size"}, axis=1)
relative_median_latency_weight_only_pt2 = weight_df.query(
    expr="base_metric == 'median_latency' & relative & runtime == 'PT2'"
).rename({"value": "relative_pt2_median_latency"}, axis=1)

relative_size_weight_only_pt2 = relative_size_weight_only_pt2[
    [
        "model_ID",
        "precision",
        "eval_run",
        "experiment_number",
        "relative_model_pt2_size",
    ]
]
relative_median_latency_weight_only_pt2 = relative_median_latency_weight_only_pt2[
    [
        "model_ID",
        "precision",
        "eval_run",
        "experiment_number",
        "relative_pt2_median_latency",
    ]
]

pareto_front_df = pd.merge(
    relative_mae_weight_only,
    relative_size_weight_only_pt2,
    on=["model_ID", "precision", "eval_run", "experiment_number"],
    how="left",
)
pareto_front_df = pd.merge(
    pareto_front_df,
    relative_median_latency_weight_only_pt2,
    on=["model_ID", "precision", "eval_run", "experiment_number"],
    how="left",
)
print(pareto_front_df.info())

pareto_scatter_3d = px.scatter_3d(
    data_frame=pareto_front_df,
    x="relative_MAE",
    y="relative_model_pt2_size",
    z="relative_pt2_median_latency",
    color="precision",
    labels=dict(
        relative_MAE="Mean Absolute Error,<br>Relative to Baseline",
        relative_model_pt2_size="PT2 Model Size,<br>Relative to Baseline",
        relative_pt2_median_latency="Median Latency,<br>Relative to Baseline",
        precision="Data Type of<br>Weight-Only PTQ",
    ),
    hover_name="model_ID",
    hover_data=[
        "hidden_layer_1_neurons",
        "hidden_layer_2_neurons",
        "hidden_layer_3_neurons",
        "activation",
    ],
    title="Pareto Frontier of Quantized Feedforward Networks",
)
pareto_scatter_3d.update_layout(
    template=font_size_template,
    margin=dict(l=120),
)
pareto_scatter_3d.update_annotations(font=dict(size=26))
pareto_scatter_3d.write_html(OUTPUT_FIGURE_PATH / "pareto_front_scatter_3d.html")

pareto_front_summary_df = (
    pareto_front_df[
        [
            "hidden_layer_1_neurons",
            "hidden_layer_2_neurons",
            "hidden_layer_3_neurons",
            "total_hidden_neurons",
            "activation",
            "precision",
            "relative_MAE",
            "relative_model_pt2_size",
            "relative_pt2_median_latency",
        ]
    ]
    .groupby(
        [
            "hidden_layer_1_neurons",
            "hidden_layer_2_neurons",
            "hidden_layer_3_neurons",
            "total_hidden_neurons",
            "activation",
            "precision",
        ]
    )
    .agg(func=["mean", "std"])
    .reset_index()
)
pareto_front_summary_df.columns = [
    "_".join(col).strip() for col in pareto_front_summary_df.columns
]
pareto_front_summary_df = pareto_front_summary_df.rename({"hidden_layer_1_neurons_":"hidden_layer_1_neurons",
                                                          "hidden_layer_2_neurons_":"hidden_layer_2_neurons",
                                                          "hidden_layer_3_neurons_":"hidden_layer_3_neurons",
                                                          "activation_":"activation",
                                                          "precision_":"precision",
                                                          "total_hidden_neurons_": "total_hidden_neurons"}, axis=1)
print(pareto_front_summary_df.info())

pareto_mae_size = px.scatter(
    pareto_front_summary_df,
    x="relative_model_pt2_size_mean",
    y="relative_MAE_mean",
    error_x="relative_model_pt2_size_std",
    error_y="relative_MAE_std",
    color="precision",
    hover_data=[
        "hidden_layer_1_neurons",
        "hidden_layer_2_neurons",
        "hidden_layer_3_neurons",
        "activation",
    ],
    labels=dict(
        relative_model_pt2_size_mean = "PT2 Model Size,<br>Relative to Baseline",
        relative_MAE_mean = "Mean Absolute Error,<br>Relative to Baseline",
        precision = "Data Type of<br>Weight-Only PTQ"
    ),
    title="Compression-Error Pareto Frontier of<br>Quantized Feedforward Networks",
)

pareto_mae_size.update_layout(template=font_size_template, margin=dict(l=140, r=120, b=120))
pareto_mae_size.update_annotations(font=dict(size=26))
pareto_mae_size.write_html(
    OUTPUT_FIGURE_PATH / "pareto_mae_size.html"
)

pareto_mae_latency = px.scatter(
    pareto_front_summary_df,
    x="relative_pt2_median_latency_mean",
    y="relative_MAE_mean",
    error_x="relative_pt2_median_latency_std",
    error_y="relative_MAE_std",
    color="precision",
    hover_data=[
        "hidden_layer_1_neurons",
        "hidden_layer_2_neurons",
        "hidden_layer_3_neurons",
        "activation",
    ],
    labels=dict(
        relative_pt2_median_latency_mean = "Median Latency,<br>Relative to Baseline",
        relative_MAE_mean = "Mean Absolute Error,<br>Relative to Baseline",
        precision = "Data Type of<br>Weight-Only PTQ"
    ),
    title="Latency-Error Pareto Frontier of<br>Quantized Feedforward Networks",
)

pareto_mae_latency.update_layout(template=font_size_template, margin=dict(l=140, r=120, b=120))
pareto_mae_latency.update_annotations(font=dict(size=26))
pareto_mae_latency.write_html(
    OUTPUT_FIGURE_PATH / "pareto_mae_latency.html"
)

relative_size_total_neurons = px.scatter(
    pareto_front_summary_df,
    x="total_hidden_neurons",
    y="relative_model_pt2_size_mean",
    error_y="relative_model_pt2_size_std",
    color="precision",
    hover_data=[
        "hidden_layer_1_neurons",
        "hidden_layer_2_neurons",
        "hidden_layer_3_neurons",
        "activation",
    ],
    labels=dict(
        relative_model_pt2_size_mean = "PT2 Model Size,<br>Relative to Baseline",
        total_hidden_neurons = "Total Number of Hidden Neurons",
        precision = "Data Type of<br>Weight-Only PTQ"
    ),
    title="Compression Ratio Versus Hidden Neurons<br>in Quantized Feedforward Networks",
)

relative_size_total_neurons.update_layout(template=font_size_template, margin=dict(l=140, r=120, b=120))
relative_size_total_neurons.update_annotations(font=dict(size=26))
relative_size_total_neurons.write_html(
    OUTPUT_FIGURE_PATH / "relative_size_total_neurons.html"
)

mae_vs_activation = px.violin(
    pareto_front_df,
    x="activation",
    y="relative_MAE",
    color="precision",
    points="all",
    labels=dict(
        relative_MAE = "Mean Absolute Error,<br>Relative to Baseline",
        activation = "Activation Function",
        precision = "Data Type of<br>Weight-Only PTQ"
    ),
    title="Relative Error Versus Activation Functions<br>in Quantized Feedforward Networks",
)
mae_vs_activation.update_layout(template=font_size_template, margin=dict(l=120))
mae_vs_activation.update_annotations(font=dict(size=26))
mae_vs_activation.write_html(
    OUTPUT_FIGURE_PATH / "mae_vs_activation.html"
)

relative_mae_total_neurons = px.scatter(
    pareto_front_summary_df,
    x="total_hidden_neurons",
    y="relative_MAE_mean",
    error_y="relative_MAE_std",
    color="precision",
    hover_data=[
        "hidden_layer_1_neurons",
        "hidden_layer_2_neurons",
        "hidden_layer_3_neurons",
        "activation",
    ],
    labels=dict(
        relative_MAE_mean = "Mean Absolute Error,<br>Relative to Baseline",
        total_hidden_neurons = "Total Number of Hidden Neurons",
        precision = "Data Type of<br>Weight-Only PTQ"
    ),
    title="Mean Absolute Error Versus Hidden Neurons<br>in Quantized Feedforward Networks",
)

relative_mae_total_neurons.update_layout(template=font_size_template, margin=dict(l=140, r=120, b=120))
relative_mae_total_neurons.update_annotations(font=dict(size=26))
relative_mae_total_neurons.write_html(
    OUTPUT_FIGURE_PATH / "relative_mae_total_neurons.html"
)

relative_median_latency_total_neurons = px.scatter(
    pareto_front_summary_df,
    x="total_hidden_neurons",
    y="relative_pt2_median_latency_mean",
    error_y="relative_pt2_median_latency_std",
    color="precision",
    hover_data=[
        "hidden_layer_1_neurons",
        "hidden_layer_2_neurons",
        "hidden_layer_3_neurons",
        "activation",
    ],
    labels=dict(
        relative_pt2_median_latency_mean = "Median PT2 Latency,<br>Relative to Baseline",
        total_hidden_neurons = "Total Number of Hidden Neurons",
        precision = "Data Type of<br>Weight-Only PTQ"
    ),
    title="Median PT2 Latency Versus Hidden Neurons<br>in Quantized Feedforward Networks",
)

relative_median_latency_total_neurons.update_layout(template=font_size_template, margin=dict(l=120))
relative_median_latency_total_neurons.update_annotations(font=dict(size=26))
relative_median_latency_total_neurons.write_html(
    OUTPUT_FIGURE_PATH / "relative_median_latency_total_neurons.html"
)

latency_vs_activation = px.violin(
    pareto_front_df,
    x="activation",
    y="relative_pt2_median_latency",
    color="precision",
    points="all",
    labels=dict(
        relative_pt2_median_latency = "Median PT2 Latency,<br>Relative to Baseline",
        activation = "Activation Function",
        precision = "Data Type of<br>Weight-Only PTQ"
    ),
    title="Relative PT2 Median Latency Versus Activation Functions<br>in Quantized Feedforward Networks",
)
latency_vs_activation.update_layout(template=font_size_template, margin=dict(l=120))
latency_vs_activation.update_annotations(font=dict(size=26))
latency_vs_activation.write_html(
    OUTPUT_FIGURE_PATH / "latency_vs_activation.html"
)

size_vs_activation = px.violin(
    pareto_front_df,
    x="activation",
    y="relative_model_pt2_size",
    color="precision",
    points="all",
    labels=dict(
        relative_model_pt2_size = "Model Size,<br>Relative to Baseline",
        activation = "Activation Function",
        precision = "Data Type of<br>Weight-Only PTQ"
    ),
    title="Compression Ratio Versus Activation Functions<br>in Quantized Feedforward Networks",
)
size_vs_activation.update_layout(template=font_size_template, margin=dict(l=120))
size_vs_activation.update_annotations(font=dict(size=26))
size_vs_activation.write_html(
    OUTPUT_FIGURE_PATH / "size_vs_activation.html"
)