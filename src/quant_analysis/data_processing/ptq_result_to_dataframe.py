import pandas as pd


def ptq_results_to_dataframe(results):

    rows = []

    for config_name, result in results.items():
        base_row = {
            "config_name": config_name,
            "precision": result["config"]["precision"],
            "bits_per_weight": result["config"]["bits_per_weight"],
            "dynamic_calibration": result["config"]["dynamic_calibration"],
            "weight_only": result["config"]["weight_only"],
        }

        pytorch_row = {f"pytorch_{k}": v for k, v in result["pytorch_result"].items()}

        row = {**base_row, **pytorch_row}

        if "onnx_result" in result:
            row.update({f"onnx_{k}": v for k, v in result["onnx_result"].items()})

        if "pt2_result" in result:
            row.update({f"pt2_{k}": v for k, v in result["pt2_result"].items()})

        rows.append(row)

    raw_dataframe = pd.DataFrame(rows)
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
    decomposed_metric_df = melted_dataframe["metric"].str.extract(
        r"^(pytorch|onnx|pt2)_(quantized|relative)_(.*)$"
    )
    decomposed_metric_df.columns = ["runtime", "relative_metric", "base_metric"]
    decomposed_metric_df["relative"] = (
        decomposed_metric_df["relative_metric"] == "relative"
    )
    decomposed_metric_df.drop(labels=["relative_metric"], inplace=True, axis=1)
    output_df = pd.concat([melted_dataframe, decomposed_metric_df], axis=1)

    return output_df
