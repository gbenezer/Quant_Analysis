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

    return pd.DataFrame(rows)
