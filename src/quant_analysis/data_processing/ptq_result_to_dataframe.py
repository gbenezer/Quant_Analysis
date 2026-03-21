import pandas as pd


def ptq_results_to_dataframe(results):
    """
    Converts nested post-training quanitization (PTQ) results into a DataFrame.

    The input `results` dictionary is expected to contain one entry per quantization configuration.

    This function builds a wide DataFrame with one row per configuration and one column per metric. Then, it reshapes 
    the DataFrame into long format so each metric becomes its own row.

    Params:
        results (dict): a nested dictionary containing PTQ configuration and runtime results.

    Returns:
        Pandas.DataFrame: a long-form DataFrame where each row represents a single metric for one configuration/result.
    """

    rows = []

    # Build one flattened row per configuration.
    for config_name, result in results.items():
        # Store the shared configuration metadata for this test.
        base_row = {
            "config_name": config_name,
            "precision": result["config"]["precision"],
            "bits_per_weight": result["config"]["bits_per_weight"],
            "dynamic_calibration": result["config"]["dynamic_calibration"],
            "weight_only": result["config"]["weight_only"],
        }

        # Add "pytorch" metric names to distinguish names after merging.
        pytorch_row = {f"pytorch_{k}": v for k, v in result["pytorch_result"].items()}

        row = {**base_row, **pytorch_row}

        # Add ONNX metrics if they exist for this configuration.
        if "onnx_result" in result:
            row.update({f"onnx_{k}": v for k, v in result["onnx_result"].items()})

        # Add PT2 metrics if they exist for this configuration.
        if "pt2_result" in result:
            row.update({f"pt2_{k}": v for k, v in result["pt2_result"].items()})

        rows.append(row)

    # Create a wide DataFrame: one row per config, one column per metric.
    raw_dataframe = pd.DataFrame(rows)

    # Convert from wide format to long format so metrics can be analyzed uniformly.
    melted_dataframe = raw_dataframe.melt(
        id_vars=[
            "config_name",
            "precision",
            "bits_per_weight",
            "dynamic_calibration",
            "weight_only",
        ],
        var_name="metric",
        value_name="value",
    )

    # Split metric names.
    decomposed_metric_df = melted_dataframe["metric"].str.extract(
        r"^(pytorch|onnx|pt2)_(quantized|relative)_(.*)$"
    )
    decomposed_metric_df.columns = ["runtime", "relative_metric", "base_metric"]

     # Convert the metric type into a boolean flag for easier filtering.
    decomposed_metric_df["relative"] = (
        decomposed_metric_df["relative_metric"] == "relative"
    )
    decomposed_metric_df.drop(labels=["relative_metric"], inplace=True, axis=1)

    # Combine the original melted data with the parsed metric components.
    output_df = pd.concat([melted_dataframe, decomposed_metric_df], axis=1)

    return output_df
