import copy
import inspect
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
    Float8StaticActivationFloat8WeightConfig,
    Float8WeightOnlyConfig,
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
    quantize_,
)

from src.quant_analysis.metric_calculation import (
    evaluate_mae,
    evaluate_onnx_latency_and_size,
    evaluate_pt2_latency_and_size,
    evaluate_pytorch_latency_and_estimate_size,
)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(f"Using device: {device}")

config_property_mapping = {
    "Float8DynamicActivationFloat8WeightConfig": {
        "precision": "float8",
        "bits_per_weight": 8,
        "calibration": "dynamic",
        "weight_only": "no",
    },
    "Float8StaticActivationFloat8WeightConfig": {
        "precision": "float8",
        "bits_per_weight": 8,
        "calibration": "static",
        "weight_only": "no",
    },
    "Int8DynamicActivationInt8WeightConfig": {
        "precision": "int8",
        "bits_per_weight": 8,
        "calibration": "dynamic",
        "weight_only": "no",
    },
    "Int8WeightOnlyConfig": {
        "precision": "int8",
        "bits_per_weight": 8,
        "calibration": "static",
        "weight_only": "yes",
    },
    "Float8WeightOnlyConfig": {
        "precision": "float8",
        "bits_per_weight": 8,
        "calibration": "static",
        "weight_only": "yes",
    },
    "Int4WeightOnlyConfig": {
        "precision": "int4",
        "bits_per_weight": 4,
        "calibration": "static",
        "weight_only": "yes",
    },
}


def supports_step(config_cls):
    return "step" in inspect.signature(config_cls).parameters


def quantize_ptq(
    base_model: nn.Module,
    ao_config: Any,
    is_static: bool = False,
    device: str | torch.device = "cpu",
    data: Optional[DataLoader] = None,
    **kwargs,
):

    model = copy.deepcopy(base_model).to(device=device)
    model.eval()

    try:
        if is_static:
            if supports_step(ao_config):
                quantize_(model=model, config=ao_config(step="prepare"))

                with torch.no_grad():
                    if data is not None:
                        for batch in data:
                            x = batch[0] if isinstance(batch, (tuple, list)) else batch
                            x = x.to(device)
                            model(x)

                quantize_(model=model, config=ao_config(step="convert", **kwargs))

            else:
                # configs without observer flow
                quantize_(model=model, config=ao_config(**kwargs))

        else:
            quantize_(model=model, config=ao_config(**kwargs))

        return model

    except AssertionError as e:
        print(f"Skipping {ao_config.__name__}: {e}")
        return None


def run_full_ptq(
    base_model: nn.Module,
    dataloader: DataLoader,
    evaluation_device: str | torch.device = "cpu",
    batch_size: int = 128,
    latency_measurements: int = 500,
    warmup_inferences: int = 50,
    print_debug: bool = False,
):

    if print_debug:
        print("constructing models")

    # construct and quantize the models
    model_dynamic_f8a_f8w = quantize_ptq(
        base_model, Float8DynamicActivationFloat8WeightConfig, device=evaluation_device
    )

    model_static_f8a_f8w = quantize_ptq(
        base_model,
        Float8StaticActivationFloat8WeightConfig,
        is_static=True,
        data=dataloader,
        device=evaluation_device,
    )

    model_dynamic_i8a_i8w = quantize_ptq(
        base_model, Int8DynamicActivationInt8WeightConfig, device=evaluation_device
    )

    model_i8w = quantize_ptq(
        base_model, Int8WeightOnlyConfig, device=evaluation_device, version=2
    )
    model_f8w = quantize_ptq(
        base_model, Float8WeightOnlyConfig, device=evaluation_device
    )
    model_i4w = quantize_ptq(base_model, Int4WeightOnlyConfig, device=evaluation_device)

    model_config_name_bit_list = [
        (model_dynamic_f8a_f8w, "Float8DynamicActivationFloat8WeightConfig", 8),
        (model_static_f8a_f8w, "Float8StaticActivationFloat8WeightConfig", 8),
        (model_dynamic_i8a_i8w, "Int8DynamicActivationInt8WeightConfig", 8),
        (model_i8w, "Int8WeightOnlyConfig", 8),
        (model_f8w, "Float8WeightOnlyConfig", 8),
        (model_i4w, "Int4WeightOnlyConfig", 4),
    ]

    # get the attributes of each config as a new dictionary
    output_dict = {}

    if print_debug:
        print("evaluating baseline MAE")

    baseline_MAE = evaluate_mae(model=base_model, dataloader=dataloader)

    if print_debug:
        print(f"Baseline MAE: {baseline_MAE}")

    input_dim = next(iter(dataloader))[0].shape[1]
    sample_input = torch.randn(
        batch_size,
        input_dim,
        device=evaluation_device,
        dtype=next(base_model.parameters()).dtype,
    )

    if print_debug:
        print(f"sample input shape: {sample_input.shape}")
        print("evaluating baseline latency and model size")

    (
        baseline_model_size,
        baseline_median_latency,
        baseline_p95_latency,
        baseline_p99_latency,
    ) = evaluate_pytorch_latency_and_estimate_size(
        base_model,
        sample_input,
        bits_per_weight=32,
        runs=latency_measurements,
        warmup=warmup_inferences,
        device=evaluation_device,
    )

    if print_debug:
        print(f"starting model quantization")

    for quantized_model, config_name, bits_per_weight in model_config_name_bit_list:
        print(f"Current config: {config_name}")

        if quantized_model is None:
            continue

        metric_dict = {}

        # evaluate model error
        if print_debug:
            print(f"evaluating {config_name} MAE")

        metric_dict["quantized_MAE"] = evaluate_mae(
            model=quantized_model, dataloader=dataloader
        )
        metric_dict["relative_MAE"] = metric_dict["quantized_MAE"] / baseline_MAE

        # evaluate model size and latency
        try:
            if print_debug:
                print(f"evaluating size and latency for {config_name}")
            (
                metric_dict["quantized_model_size"],
                metric_dict["quantized_median_latency"],
                metric_dict["quantized_p95_latency"],
                metric_dict["quantized_p99_latency"],
            ) = evaluate_pytorch_latency_and_estimate_size(
                quantized_model,
                sample_input,
                bits_per_weight=bits_per_weight,
                runs=latency_measurements,
                warmup=warmup_inferences,
                device=evaluation_device,
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

            output_dict[config_name] = copy.deepcopy(
                config_property_mapping[config_name]
            )
            output_dict[config_name].update(metric_dict)

        except Exception as e:
            print(f"Skipping export for {config_name}: {e}")
            continue

    return output_dict


def run_weight_only_ptq(
    base_model: nn.Module,
    dataloader: DataLoader,
    evaluation_device: str | torch.device = "cpu",
    batch_size: int = 128,
    latency_measurements: int = 500,
    warmup_inferences: int = 50,
    print_debug: bool = False,
):

    if print_debug:
        print("constructing models")

    # construct and quantize the models
    model_i8w = quantize_ptq(
        base_model, Int8WeightOnlyConfig, device=evaluation_device, version=2
    )
    model_f8w = quantize_ptq(
        base_model, Float8WeightOnlyConfig, device=evaluation_device
    )
    model_i4w = quantize_ptq(base_model, Int4WeightOnlyConfig, device=evaluation_device)

    model_config_name_bit_list = [
        (model_i8w, "Int8WeightOnlyConfig", 8),
        (model_f8w, "Float8WeightOnlyConfig", 8),
        (model_i4w, "Int4WeightOnlyConfig", 4),
    ]

    # get the attributes of each config as a new dictionary
    pytorch_output_dict = {}
    onnx_output_dict = {}
    pt2_output_dict = {}

    if print_debug:
        print("evaluating baseline MAE")

    baseline_MAE = evaluate_mae(model=base_model, dataloader=dataloader)

    if print_debug:
        print(f"Baseline MAE: {baseline_MAE}")

    input_dim = next(iter(dataloader))[0].shape[1]
    sample_input = torch.randn(
        batch_size,
        input_dim,
        device=evaluation_device,
        dtype=next(base_model.parameters()).dtype,
    )

    if print_debug:
        print(f"sample input shape: {sample_input.shape}")
        print("evaluating baseline latency and model size")

    # (
    #     baseline_model_size_pytorch,
    #     baseline_median_latency_pytorch,
    #     baseline_p95_latency_pytorch,
    #     baseline_p99_latency_pytorch,
    # )
    pytorch_size_latency_results = evaluate_pytorch_latency_and_estimate_size(
        base_model,
        sample_input,
        bits_per_weight=32,
        runs=latency_measurements,
        warmup=warmup_inferences,
        device=evaluation_device,
    )

    # (
    #     baseline_model_size_onnx,
    #     baseline_median_latency_onnx,
    #     baseline_p95_latency_onnx,
    #     baseline_p99_latency_onnx,
    # )
    onnx_size_latency_results = evaluate_onnx_latency_and_size(
        base_model,
        input_dim,
        latency_measurements=latency_measurements,
        warmup_inferences=warmup_inferences,
        device=evaluation_device,
    )

    # (
    #     baseline_model_size_pt2,
    #     baseline_median_latency_pt2,
    #     baseline_p95_latency_pt2,
    #     baseline_p99_latency_pt2,
    # )
    pt2_size_latency_results = evaluate_pt2_latency_and_size(
        base_model,
        input_dim,
        num_runs=latency_measurements,
        warmup_runs=warmup_inferences,
        device=evaluation_device,
    )

    if print_debug:
        print(f"starting model quantization")

    for quantized_model, config_name, bits_per_weight in model_config_name_bit_list:
        print(f"Current config: {config_name}")

        if quantized_model is None:
            continue

        pytorch_metric_dict = {}
        onnx_metric_dict = {}
        pt2_metric_dict = {}

        # pytorch
        # evaluate model error
        if print_debug:
            print(f"evaluating {config_name} MAE")

        pytorch_metric_dict["quantized_MAE"] = evaluate_mae(
            model=quantized_model, dataloader=dataloader
        )
        pytorch_metric_dict["relative_MAE"] = (
            pytorch_metric_dict["quantized_MAE"] / baseline_MAE
        )

        # evaluate model size and latency
        try:
            if print_debug:
                print(f"evaluating size and latency for {config_name}")
            (
                pytorch_metric_dict["quantized_model_size"],
                pytorch_metric_dict["quantized_median_latency"],
                pytorch_metric_dict["quantized_p95_latency"],
                pytorch_metric_dict["quantized_p99_latency"],
            ) = evaluate_pytorch_latency_and_estimate_size(
                quantized_model,
                sample_input,
                bits_per_weight=bits_per_weight,
                runs=latency_measurements,
                warmup=warmup_inferences,
                device=evaluation_device,
            )

            pytorch_metric_dict["relative_model_size"] = (
                pytorch_metric_dict["quantized_model_size"]
                / pytorch_size_latency_results[0]
            )
            pytorch_metric_dict["relative_median_latency"] = (
                pytorch_metric_dict["quantized_median_latency"]
                / pytorch_size_latency_results[1]
            )
            pytorch_metric_dict["relative_p95_latency"] = (
                pytorch_metric_dict["quantized_p95_latency"]
                / pytorch_size_latency_results[2]
            )
            pytorch_metric_dict["relative_p99_latency"] = (
                pytorch_metric_dict["quantized_p99_latency"]
                / pytorch_size_latency_results[3]
            )

            pytorch_output_dict[config_name] = copy.deepcopy(
                config_property_mapping[config_name]
            )
            pytorch_output_dict[config_name].update(pytorch_metric_dict)

        except Exception as e:
            print(f"Skipping export for {config_name}: {e}")
            continue

        # onnx
        # evaluate model size and latency
        try:
            if print_debug:
                print(f"evaluating size and latency for {config_name}, ONNX")
            (
                onnx_metric_dict["quantized_model_size"],
                onnx_metric_dict["quantized_median_latency"],
                onnx_metric_dict["quantized_p95_latency"],
                onnx_metric_dict["quantized_p99_latency"],
            ) = evaluate_onnx_latency_and_size(
                quantized_model,
                input_dim=input_dim,
                latency_measurements=latency_measurements,
                warmup_inferences=warmup_inferences,
                device=evaluation_device,
            )

            onnx_metric_dict["relative_model_size"] = (
                onnx_metric_dict["quantized_model_size"] / onnx_size_latency_results[0]
            )
            onnx_metric_dict["relative_median_latency"] = (
                onnx_metric_dict["quantized_median_latency"]
                / onnx_size_latency_results[1]
            )
            onnx_metric_dict["relative_p95_latency"] = (
                onnx_metric_dict["quantized_p95_latency"] / onnx_size_latency_results[2]
            )
            onnx_metric_dict["relative_p99_latency"] = (
                onnx_metric_dict["quantized_p99_latency"] / onnx_size_latency_results[3]
            )

            onnx_output_dict[config_name] = copy.deepcopy(
                config_property_mapping[config_name]
            )
            onnx_output_dict[config_name].update(onnx_metric_dict)

        except Exception as e:
            print(f"Skipping export for {config_name}: {e}")
            continue

        # pt2
        # evaluate model size and latency
        try:
            if print_debug:
                print(f"evaluating size and latency for {config_name}, PT2")
            (
                pt2_metric_dict["quantized_model_size"],
                pt2_metric_dict["quantized_median_latency"],
                pt2_metric_dict["quantized_p95_latency"],
                pt2_metric_dict["quantized_p99_latency"],
            ) = evaluate_pt2_latency_and_size(
                quantized_model,
                input_dim=input_dim,
                num_runs=latency_measurements,
                warmup_runs=warmup_inferences,
                device=evaluation_device,
            )

            pt2_metric_dict["relative_model_size"] = (
                pt2_metric_dict["quantized_model_size"] / pt2_size_latency_results[0]
            )
            pt2_metric_dict["relative_median_latency"] = (
                pt2_metric_dict["quantized_median_latency"]
                / pt2_size_latency_results[1]
            )
            pt2_metric_dict["relative_p95_latency"] = (
                pt2_metric_dict["quantized_p95_latency"] / pt2_size_latency_results[2]
            )
            pt2_metric_dict["relative_p99_latency"] = (
                pt2_metric_dict["quantized_p99_latency"] / pt2_size_latency_results[3]
            )

            pt2_output_dict[config_name] = copy.deepcopy(
                config_property_mapping[config_name]
            )
            pt2_output_dict[config_name].update(pt2_metric_dict)

        except Exception as e:
            print(f"Skipping export for {config_name}: {e}")
            continue

    return {
        "pytorch": pytorch_output_dict,
        "onnx": onnx_output_dict,
        "pt2": pt2_output_dict,
    }


if __name__ == "__main__":
    from data.load_data import get_superconductivity_data
    from src.quant_analysis.model_loading import load_mlp_from_pth

    print("loading model")
    test_model = load_mlp_from_pth(
        path=(Path.cwd() / "models" / "state_dicts" / "base_model_FP32.pth")
    ).to(device=device)

    print("getting data")
    (
        _,
        _,
        _,
        _,
        train_loader,
        _,
        test_loader,
    ) = get_superconductivity_data(
        test_fraction=0.2, random_seed=12, n_workers=4, batch_n=32
    )

    print("running ptq")
    train_loader_output = run_full_ptq(
        test_model, train_loader, evaluation_device=device, print_debug=True
    )

    # test_loader_output = run_ptq(test_model, test_loader, device=device)

    print(train_loader_output)
    # print(test_loader_output)
