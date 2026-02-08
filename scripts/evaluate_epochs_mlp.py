from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px

LOG_DIRECTORY = Path("models/logs/")

# load base model data
base_fp32_df = pd.read_csv(
    filepath_or_buffer=(
        LOG_DIRECTORY
        / "base_model_FP32"
        / "base_model_FP32_csv_log"
        / "version_0"
        / "metrics.csv"
    )
)
base_fp64_df = pd.read_csv(
    filepath_or_buffer=(
        LOG_DIRECTORY
        / "base_model_FP64"
        / "base_model_FP64_csv_log"
        / "version_0"
        / "metrics.csv"
    )
)
base_fp32_no_norm_df = pd.read_csv(
    filepath_or_buffer=(
        LOG_DIRECTORY
        / "base_model_FP32_no_norm"
        / "base_model_FP32_no_norm_csv_log"
        / "version_0"
        / "metrics.csv"
    )
)
base_fp64_no_norm_df = pd.read_csv(
    filepath_or_buffer=(
        LOG_DIRECTORY
        / "base_model_FP64_no_norm"
        / "base_model_FP64_no_norm_csv_log"
        / "version_0"
        / "metrics.csv"
    )
)

# select relevant data portions for concatenation
base_fp32_valid_df = (
    base_fp32_df.loc[:, ["epoch", "step", "valid_loss"]]
    .dropna(axis=0, how="any")
    .assign(model="base_fp32")
)
base_fp64_valid_df = (
    base_fp64_df.loc[:, ["epoch", "step", "valid_loss"]]
    .dropna(axis=0, how="any")
    .assign(model="base_fp64")
)
base_fp32_no_norm_valid_df = (
    base_fp32_no_norm_df.loc[:, ["epoch", "step", "valid_loss"]]
    .dropna(axis=0, how="any")
    .assign(model="base_fp32_no_norm")
)
base_fp64_no_norm_valid_df = (
    base_fp64_no_norm_df.loc[:, ["epoch", "step", "valid_loss"]]
    .dropna(axis=0, how="any")
    .assign(model="base_fp64_no_norm")
)

# merge the DataFrames
validation_data = pd.concat(
    [
        base_fp32_valid_df,
        base_fp64_valid_df,
        base_fp32_no_norm_valid_df,
        base_fp64_no_norm_valid_df,
    ]
)

figure = px.scatter(data_frame=validation_data, x="epoch", y="valid_loss", color="model")
figure.write_html("scripts/validation_loss.html")
