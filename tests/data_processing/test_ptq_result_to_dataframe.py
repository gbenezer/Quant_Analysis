import pytest
import pandas as pd

from src.quant_analysis.data_processing.ptq_result_to_dataframe import ptq_results_to_dataframe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = {
    "config_name", "precision", "bits_per_weight", "dynamic_calibration",
    "weight_only", "metric", "value", "runtime", "base_metric", "relative",
}


def make_entry(
    config_name,
    *,
    precision="fp32",
    bits=32,
    dynamic=False,
    weight_only=False,
    pytorch_metrics=None,
    onnx_metrics=None,
    pt2_metrics=None,
):
    """Build a single-key results dict suitable for passing to ptq_results_to_dataframe."""
    entry = {
        "config": {
            "precision": precision,
            "bits_per_weight": bits,
            "dynamic_calibration": dynamic,
            "weight_only": weight_only,
        },
        "pytorch_result": pytorch_metrics or {"quantized_mae": 0.10, "relative_mae": 0.00},
    }
    if onnx_metrics is not None:
        entry["onnx_result"] = onnx_metrics
    if pt2_metrics is not None:
        entry["pt2_result"] = pt2_metrics
    return {config_name: entry}


def merge(*dicts):
    result = {}
    for d in dicts:
        result.update(d)
    return result


def get_single_row(df, metric):
    """Return the unique row matching `metric`; fail if there is not exactly one."""
    rows = df[df["metric"] == metric]
    assert len(rows) == 1, f"Expected exactly one row with metric='{metric}', got {len(rows)}"
    return rows.iloc[0]


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_dataframe(self):
        result = ptq_results_to_dataframe(make_entry("cfg"))
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Output columns
# ---------------------------------------------------------------------------

class TestOutputColumns:
    def test_all_expected_columns_present(self):
        result = ptq_results_to_dataframe(make_entry("cfg"))
        assert EXPECTED_COLUMNS.issubset(set(result.columns))

    def test_relative_metric_intermediate_column_dropped(self):
        result = ptq_results_to_dataframe(make_entry("cfg"))
        assert "relative_metric" not in result.columns

    def test_no_extra_columns_for_single_runtime(self):
        result = ptq_results_to_dataframe(make_entry("cfg"))
        assert set(result.columns) == EXPECTED_COLUMNS


# ---------------------------------------------------------------------------
# Row count
# ---------------------------------------------------------------------------

class TestRowCount:
    def test_single_config_pytorch_only(self):
        data = make_entry("cfg", pytorch_metrics={"quantized_mae": 0.1, "relative_mae": 0.0})
        assert len(ptq_results_to_dataframe(data)) == 2

    def test_single_config_pytorch_and_onnx(self):
        data = make_entry(
            "cfg",
            pytorch_metrics={"quantized_mae": 0.10, "relative_mae": 0.00},
            onnx_metrics={"quantized_mae": 0.11, "relative_mae": 0.01},
        )
        assert len(ptq_results_to_dataframe(data)) == 4

    def test_single_config_all_three_runtimes(self):
        data = make_entry(
            "cfg",
            pytorch_metrics={"quantized_mae": 0.10, "relative_mae": 0.00},
            onnx_metrics={"quantized_mae": 0.11, "relative_mae": 0.01},
            pt2_metrics={"quantized_mae": 0.12, "relative_mae": 0.02},
        )
        assert len(ptq_results_to_dataframe(data)) == 6

    def test_two_configs_pytorch_only(self):
        data = merge(
            make_entry("cfg_a", pytorch_metrics={"quantized_mae": 0.1, "relative_mae": 0.0}),
            make_entry("cfg_b", pytorch_metrics={"quantized_mae": 0.2, "relative_mae": 0.1}),
        )
        assert len(ptq_results_to_dataframe(data)) == 4

    def test_two_configs_both_with_onnx(self):
        data = merge(
            make_entry(
                "cfg_a",
                pytorch_metrics={"quantized_mae": 0.10, "relative_mae": 0.00},
                onnx_metrics={"quantized_mae": 0.11, "relative_mae": 0.01},
            ),
            make_entry(
                "cfg_b",
                pytorch_metrics={"quantized_mae": 0.20, "relative_mae": 0.10},
                onnx_metrics={"quantized_mae": 0.21, "relative_mae": 0.11},
            ),
        )
        assert len(ptq_results_to_dataframe(data)) == 8

    def test_multiple_metrics_per_runtime_scale_row_count(self):
        pytorch_metrics = {
            "quantized_mae": 0.1,
            "relative_mae": 0.0,
            "quantized_latency_ms": 50.0,
            "relative_latency_ms": 0.2,
        }
        data = make_entry("cfg", pytorch_metrics=pytorch_metrics)
        assert len(ptq_results_to_dataframe(data)) == 4


# ---------------------------------------------------------------------------
# Metric parsing — runtime, base_metric, relative
# ---------------------------------------------------------------------------

class TestMetricParsing:
    def test_pytorch_runtime_extracted(self):
        data = make_entry("cfg", pytorch_metrics={"quantized_mae": 0.1, "relative_mae": 0.0})
        row = get_single_row(ptq_results_to_dataframe(data), "pytorch_quantized_mae")
        assert row["runtime"] == "pytorch"

    def test_onnx_runtime_extracted(self):
        data = make_entry(
            "cfg",
            pytorch_metrics={"quantized_mae": 0.10, "relative_mae": 0.00},
            onnx_metrics={"quantized_mae": 0.11, "relative_mae": 0.01},
        )
        row = get_single_row(ptq_results_to_dataframe(data), "onnx_quantized_mae")
        assert row["runtime"] == "onnx"

    def test_pt2_runtime_extracted(self):
        data = make_entry(
            "cfg",
            pytorch_metrics={"quantized_mae": 0.10, "relative_mae": 0.00},
            pt2_metrics={"quantized_mae": 0.12, "relative_mae": 0.02},
        )
        row = get_single_row(ptq_results_to_dataframe(data), "pt2_quantized_mae")
        assert row["runtime"] == "pt2"

    def test_base_metric_extracted(self):
        data = make_entry("cfg", pytorch_metrics={"quantized_mae": 0.1, "relative_mae": 0.0})
        row = get_single_row(ptq_results_to_dataframe(data), "pytorch_quantized_mae")
        assert row["base_metric"] == "mae"

    def test_quantized_prefix_sets_relative_false(self):
        data = make_entry("cfg", pytorch_metrics={"quantized_mae": 0.1, "relative_mae": 0.0})
        row = get_single_row(ptq_results_to_dataframe(data), "pytorch_quantized_mae")
        assert row["relative"] == False

    def test_relative_prefix_sets_relative_true(self):
        data = make_entry("cfg", pytorch_metrics={"quantized_mae": 0.1, "relative_mae": 0.0})
        row = get_single_row(ptq_results_to_dataframe(data), "pytorch_relative_mae")
        assert row["relative"] == True

    def test_multipart_base_metric_name_preserved(self):
        # Underscores inside the base metric name should be kept intact.
        data = make_entry(
            "cfg",
            pytorch_metrics={"quantized_model_size_mb": 10.0, "relative_model_size_mb": 0.5},
        )
        row = get_single_row(ptq_results_to_dataframe(data), "pytorch_quantized_model_size_mb")
        assert row["base_metric"] == "model_size_mb"

    @pytest.mark.parametrize("runtime", ["pytorch", "onnx", "pt2"])
    def test_all_runtimes_parsed_correctly(self, runtime):
        metrics = {"quantized_mae": 0.1, "relative_mae": 0.0}
        kwargs = {f"{runtime}_metrics": metrics}
        if runtime != "pytorch":
            kwargs["pytorch_metrics"] = {"quantized_mae": 0.1, "relative_mae": 0.0}
        data = make_entry("cfg", **kwargs)
        result = ptq_results_to_dataframe(data)
        runtime_rows = result[result["runtime"] == runtime]
        assert len(runtime_rows) > 0


# ---------------------------------------------------------------------------
# Config metadata propagation
# ---------------------------------------------------------------------------

class TestConfigMetadata:
    def test_config_name_propagated_to_all_rows(self):
        result = ptq_results_to_dataframe(make_entry("my_config"))
        assert (result["config_name"] == "my_config").all()

    def test_precision_propagated(self):
        result = ptq_results_to_dataframe(make_entry("cfg", precision="int8"))
        assert (result["precision"] == "int8").all()

    def test_bits_per_weight_propagated(self):
        result = ptq_results_to_dataframe(make_entry("cfg", bits=8))
        assert (result["bits_per_weight"] == 8).all()

    def test_dynamic_calibration_true_propagated(self):
        result = ptq_results_to_dataframe(make_entry("cfg", dynamic=True))
        assert (result["dynamic_calibration"] == True).all()

    def test_weight_only_true_propagated(self):
        result = ptq_results_to_dataframe(make_entry("cfg", weight_only=True))
        assert (result["weight_only"] == True).all()

    def test_multiple_configs_retain_distinct_names(self):
        data = merge(
            make_entry("cfg_a", pytorch_metrics={"quantized_mae": 0.1, "relative_mae": 0.0}),
            make_entry("cfg_b", pytorch_metrics={"quantized_mae": 0.2, "relative_mae": 0.1}),
        )
        result = ptq_results_to_dataframe(data)
        assert set(result["config_name"].unique()) == {"cfg_a", "cfg_b"}

    def test_multiple_configs_distinct_metadata_preserved(self):
        data = merge(
            make_entry("cfg_fp32", precision="fp32", bits=32),
            make_entry("cfg_int8", precision="int8", bits=8),
        )
        result = ptq_results_to_dataframe(data)
        assert result.loc[result["config_name"] == "cfg_fp32", "precision"].iloc[0] == "fp32"
        assert result.loc[result["config_name"] == "cfg_int8", "precision"].iloc[0] == "int8"


# ---------------------------------------------------------------------------
# Metric values
# ---------------------------------------------------------------------------

class TestMetricValues:
    def test_pytorch_quantized_value_correct(self):
        data = make_entry("cfg", pytorch_metrics={"quantized_mae": 0.42, "relative_mae": 0.0})
        result = ptq_results_to_dataframe(data)
        row = get_single_row(result, "pytorch_quantized_mae")
        assert float(row["value"]) == pytest.approx(0.42)

    def test_pytorch_relative_value_correct(self):
        data = make_entry("cfg", pytorch_metrics={"quantized_mae": 0.1, "relative_mae": 0.07})
        result = ptq_results_to_dataframe(data)
        row = get_single_row(result, "pytorch_relative_mae")
        assert float(row["value"]) == pytest.approx(0.07)

    def test_onnx_value_correct(self):
        data = make_entry(
            "cfg",
            pytorch_metrics={"quantized_mae": 0.10, "relative_mae": 0.00},
            onnx_metrics={"quantized_mae": 0.99, "relative_mae": 0.05},
        )
        result = ptq_results_to_dataframe(data)
        row = get_single_row(result, "onnx_quantized_mae")
        assert float(row["value"]) == pytest.approx(0.99)

    def test_pt2_value_correct(self):
        data = make_entry(
            "cfg",
            pytorch_metrics={"quantized_mae": 0.10, "relative_mae": 0.00},
            pt2_metrics={"quantized_mae": 0.55, "relative_mae": 0.03},
        )
        result = ptq_results_to_dataframe(data)
        row = get_single_row(result, "pt2_quantized_mae")
        assert float(row["value"]) == pytest.approx(0.55)


# ---------------------------------------------------------------------------
# Optional runtimes
# ---------------------------------------------------------------------------

class TestOptionalRuntimes:
    def test_no_onnx_rows_when_onnx_result_absent(self):
        data = make_entry("cfg", pytorch_metrics={"quantized_mae": 0.1, "relative_mae": 0.0})
        result = ptq_results_to_dataframe(data)
        assert len(result[result["metric"].str.startswith("onnx_")]) == 0

    def test_no_pt2_rows_when_pt2_result_absent(self):
        data = make_entry("cfg", pytorch_metrics={"quantized_mae": 0.1, "relative_mae": 0.0})
        result = ptq_results_to_dataframe(data)
        assert len(result[result["metric"].str.startswith("pt2_")]) == 0

    def test_onnx_rows_present_when_onnx_result_provided(self):
        data = make_entry(
            "cfg",
            pytorch_metrics={"quantized_mae": 0.10, "relative_mae": 0.00},
            onnx_metrics={"quantized_mae": 0.11, "relative_mae": 0.01},
        )
        result = ptq_results_to_dataframe(data)
        assert len(result[result["metric"].str.startswith("onnx_")]) == 2

    def test_pt2_rows_present_when_pt2_result_provided(self):
        data = make_entry(
            "cfg",
            pytorch_metrics={"quantized_mae": 0.10, "relative_mae": 0.00},
            pt2_metrics={"quantized_mae": 0.12, "relative_mae": 0.02},
        )
        result = ptq_results_to_dataframe(data)
        assert len(result[result["metric"].str.startswith("pt2_")]) == 2

    def test_mixed_configs_nan_value_for_missing_onnx(self):
        # cfg_a has onnx_result; cfg_b does not — cfg_b's onnx metric rows should be NaN.
        data = merge(
            make_entry(
                "cfg_a",
                pytorch_metrics={"quantized_mae": 0.10, "relative_mae": 0.00},
                onnx_metrics={"quantized_mae": 0.11, "relative_mae": 0.01},
            ),
            make_entry(
                "cfg_b",
                pytorch_metrics={"quantized_mae": 0.20, "relative_mae": 0.10},
            ),
        )
        result = ptq_results_to_dataframe(data)
        cfg_b_onnx = result[
            (result["config_name"] == "cfg_b") & result["metric"].str.startswith("onnx_")
        ]
        assert cfg_b_onnx["value"].isna().all()
