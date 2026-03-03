import copy
import tempfile
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
    Float8StaticActivationFloat8WeightConfig,
    Float8WeightOnlyConfig,
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig,
    Int8StaticActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
    quantize_,
)

from src.quant_analysis.metric_calculation import (
    evaluate_mae,
    evaluate_onnx_latency_and_size,
)

config_property_mapping = {
    "Float8DynamicActivationFloat8WeightConfig": {
        "precision": "float8",
        "calibration": "dynamic",
        "weight_only": "no",
    },
    "Float8StaticActivationFloat8WeightConfig": {
        "precision": "float8",
        "calibration": "static",
        "weight_only": "no",
    },
    "Int8DynamicActivationInt8WeightConfig": {
        "precision": "int8",
        "calibration": "dynamic",
        "weight_only": "no",
    },
    "Int8StaticActivationInt8WeightConfig": {
        "precision": "int8",
        "calibration": "static",
        "weight_only": "no",
    },
    "Int8WeightOnlyConfig": {
        "precision": "int8",
        "calibration": "static",
        "weight_only": "yes",
    },
    "Float8WeightOnlyConfig": {
        "precision": "float8",
        "calibration": "static",
        "weight_only": "yes",
    },
    "Int4WeightOnlyConfig": {
        "precision": "int4",
        "calibration": "static",
        "weight_only": "yes",
    },
}


def quantize_ptq(
    base_model: nn.Module,
    ao_config: Any,
    is_static: bool = False,
    data: Optional[DataLoader] = None,
):

    if is_static and data is None:
        raise ValueError("Static quantization requires calibration data")

    model = copy.deepcopy(base_model)
    model.eval()

    if is_static:
        quantize_(model=model, config=ao_config(step="prepare"))

        device = next(model.parameters()).device

        with torch.no_grad():
            if data is not None:
                for batch in data:
                    x = batch[0] if isinstance(batch, (tuple, list)) else batch
                    x = x.to(device)
                    model(x)

        quantize_(model=model, config=ao_config(step="convert"))

    else:
        quantize_(model=model, config=ao_config())

    return model


def run_ptq(
    base_model: nn.Module,
    dataloader: DataLoader,
    latency_measurements: int = 500,
    warmup_inferences: int = 50,
):

    # construct and quantize the models
    model_dynamic_f8a_f8w = quantize_ptq(
        base_model, Float8DynamicActivationFloat8WeightConfig
    )
    model_static_f8a_f8w = quantize_ptq(
        base_model,
        Float8StaticActivationFloat8WeightConfig,
        is_static=True,
        data=dataloader,
    )

    model_dynamic_i8a_i8w = quantize_ptq(
        base_model, Int8DynamicActivationInt8WeightConfig
    )
    model_static_i8a_i8w = quantize_ptq(
        base_model,
        Int8StaticActivationInt8WeightConfig,
        is_static=True,
        data=dataloader,
    )
    model_i8w = quantize_ptq(base_model, Int8WeightOnlyConfig)
    model_f8w = quantize_ptq(base_model, Float8WeightOnlyConfig)
    model_i4w = quantize_ptq(base_model, Int4WeightOnlyConfig)

    model_config_name_list = [
        (model_dynamic_f8a_f8w, "Float8DynamicActivationFloat8WeightConfig"),
        (model_static_f8a_f8w, "Float8StaticActivationFloat8WeightConfig"),
        (model_dynamic_i8a_i8w, "Int8DynamicActivationInt8WeightConfig"),
        (model_static_i8a_i8w, "Int8StaticActivationInt8WeightConfig"),
        (model_i8w, "Int8WeightOnlyConfig"),
        (model_f8w, "Float8WeightOnlyConfig"),
        (model_i4w, "Int4WeightOnlyConfig"),
    ]

    # get the attributes of each config as a new dictionary
    output_dict = copy.deepcopy(config_property_mapping)

    baseline_MAE = evaluate_mae(model=base_model, dataloader=dataloader)

    input_dim = next(iter(dataloader))[0].shape[1]

    (
        baseline_model_size,
        baseline_median_latency,
        baseline_p95_latency,
        baseline_p99_latency,
    ) = evaluate_onnx_latency_and_size(
        base_model,
        input_dim=input_dim,
        latency_measurements=latency_measurements,
        warmup_inferences=warmup_inferences,
    )

    for quantized_model, config_name in model_config_name_list:
        metric_dict = {}

        # evaluate model error
        metric_dict["quantized_MAE"] = evaluate_mae(
            model=quantized_model, dataloader=dataloader
        )
        metric_dict["relative_MAE"] = metric_dict["quantized_MAE"] / baseline_MAE

        # evaluate model size and latency
        (
            metric_dict["quantized_model_size"],
            metric_dict["quantized_median_latency"],
            metric_dict["quantized_p95_latency"],
            metric_dict["quantized_p99_latency"],
        ) = evaluate_onnx_latency_and_size(
            quantized_model,
            input_dim,
            latency_measurements=latency_measurements,
            warmup_inferences=warmup_inferences,
        )

        metric_dict["relative_model_size"] = (
            metric_dict["quantized_model_size"] / baseline_model_size
        )
        metric_dict["relative_median_latency"] = (
            metric_dict["quantized_median_latency"] / baseline_median_latency
        )
        metric_dict["relative_p95_latency"] = (
            metric_dict["quantized_p95_latency"] / baseline_p95_latency
        )
        metric_dict["relative_p99_latency"] = (
            metric_dict["quantized_p99_latency"] / baseline_p99_latency
        )

        output_dict[config_name].update(metric_dict)

    return output_dict
