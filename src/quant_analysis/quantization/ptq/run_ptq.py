from numbers import Real
from pathlib import Path
from typing import Any, Dict, NotRequired, Tuple, TypedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.quant_analysis.data_processing.ptq_result_to_dataframe import (
    ptq_results_to_dataframe,
)
from src.quant_analysis.metric_calculation import (
    assess_relative_performance,
    evaluate_mae,
    evaluate_onnx_latency_and_size,
    evaluate_pt2_latency_and_size,
    evaluate_pytorch_latency_and_estimate_size,
)
from src.quant_analysis.model_architecture import SimpleMLP
from src.quant_analysis.model_loading import load_mlp_from_pth
from src.quant_analysis.quantization.ptq.ptq_config_metadata import (
    PTQ_QUANT_CONFIG_METADATA,
    PTQ_WEIGHT_ONLY_CONFIG_METADATA,
    ConfigAndMetadataPTQ,
)
from src.quant_analysis.quantization.ptq.quantize_ptq import fuse_mlp_bn, quantize_ptq

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(f"Using device: {device}")


# helper function to construct all the quantized models
def build_quantized_models(
    base_model: nn.Module,
    configs: Dict[str, ConfigAndMetadataPTQ],
    dataloader: DataLoader,
    quantization_device: str | torch.device = "cpu",
) -> Dict[str, Tuple[nn.Module, ConfigAndMetadataPTQ]]:
    # initialize dictionary for model storage
    model_dict = {}

    for name, metadata in configs.items():
        # get whether the quantization is static or not
        current_model_is_static = not metadata["dynamic_calibration"]

        # store the quantized model if possible
        try:
            quant_model = quantize_ptq(
                base_model=base_model,
                ao_config=metadata["ao_config"],
                is_static=current_model_is_static,
                quantize_device=quantization_device,
                data=dataloader,
                **metadata["ao_config_kwargs"],
            )

            if quant_model is not None:
                model_dict[name] = (quant_model, metadata)

            else:
                print(f"Quantization failed for {name}. Config skipped.")

        # if there's a failure, make a note of it to the output
        except Exception as e:
            print(f"Skipping {name} PTQ Configuration: {e}")

    return model_dict


# Output result TypedDict
class QuantizationResult(TypedDict, total=True):
    config: ConfigAndMetadataPTQ
    pytorch_result: Dict[str, Real]

    # only for use with compatible configurations
    onnx_result: NotRequired[Dict[str, Real]]
    pt2_result: NotRequired[Dict[str, Real]]


def run_ptq(
    base_model: nn.Module,
    dataloader: DataLoader,
    evaluation_device: str | torch.device = "cpu",
    batch_size: int = 128,
    runs: int = 500,
    warmup: int = 50,
    print_debug: bool = False,
    weight_only: bool = False,
) -> Dict[str, QuantizationResult]:

    # initialize the type of the output dict
    output_dict: Dict[str, QuantizationResult] = {}

    if weight_only:
        quantization_configs = PTQ_WEIGHT_ONLY_CONFIG_METADATA
    else:
        quantization_configs = PTQ_QUANT_CONFIG_METADATA

    if print_debug:
        print(f"Weight Only: {weight_only}")
        print("Quantization Configurations Evaluated:")
        print(quantization_configs.keys())
        print("Building model dictionary")

    # build the quantized models using the helper function
    model_dictionary = build_quantized_models(
        base_model=base_model,
        configs=quantization_configs,
        dataloader=dataloader,
        quantization_device=evaluation_device,
    )

    # the quantized models are folded, but the base model needs to be folded/fused
    # in the case of the SimpleMLP
    if isinstance(base_model, SimpleMLP):
        base_model = fuse_mlp_bn(base_model)

    # get the input dimensionality of the dataset and
    # a representative sample input
    batch = next(iter(dataloader))
    sample_input = batch[0][:batch_size].to(evaluation_device)

    # get baseline evaluations
    if print_debug:
        print("Evaluating baseline model")

    try:
        baseline_MAE = evaluate_mae(model=base_model, dataloader=dataloader)
    except Exception as e:
        # if there's no way to get the MAE for the base model there's not a reason to move forward
        raise RuntimeError(
            f"The mean absolute error evaluation failed with exception {e}"
        )

    baseline_size_latency_results = {
        "pytorch": evaluate_pytorch_latency_and_estimate_size(
            model=base_model,
            sample_input=sample_input,
            device=evaluation_device,
            bits_per_weight=32,
            runs=runs,
            warmup=warmup,
        )
    }

    # only calculate other runtimes if it makes sense
    if weight_only:
        if print_debug:
            print("Evaluating ONNX and PT2 baseline performance")

        baseline_size_latency_results["onnx"] = evaluate_onnx_latency_and_size(
            model=base_model,
            sample_input=sample_input,
            device=evaluation_device,
            runs=runs,
            warmup=warmup,
        )

        baseline_size_latency_results["pt2"] = evaluate_pt2_latency_and_size(
            model=base_model,
            sample_input=sample_input,
            device=evaluation_device,
            runs=runs,
            warmup=warmup,
        )

    # iterate through the quantized models for evaluation
    if print_debug:
        print("Evaluating quantized model performances")

    for config_name, (quant_model, quant_metadata) in model_dictionary.items():
        # if the quantization failed, do not evaluate
        if quant_model is None:
            continue

        if print_debug:
            print(f"Evaluating {config_name} on PyTorch")

        # Int4WeightOnlyConfig only works with BFloat16 inputs on GPU
        if config_name == "Int4WeightOnlyConfig" and torch.device(
            evaluation_device
        ) == torch.device("cuda"):
            input_dtype = torch.bfloat16
        else:
            input_dtype = torch.float32

        # evaluate PyTorch performance
        pytorch_metric_dict: Dict[str, Any] = {}

        # MAE
        try:
            pytorch_metric_dict["quantized_MAE"] = evaluate_mae(
                model=quant_model, dataloader=dataloader, input_dtype=input_dtype
            )
            pytorch_metric_dict["relative_MAE"] = (
                pytorch_metric_dict["quantized_MAE"] / baseline_MAE
            )
        except Exception as e:
            # if the MAE measurement fails, there's no point in continuing evaluation of the configuration
            print(f"Skipping {config_name}: {e}")
            continue

        # PyTorch size and latency
        # get the results
        quantized_pytorch_size_latency = evaluate_pytorch_latency_and_estimate_size(
            model=quant_model,
            sample_input=sample_input,
            device=evaluation_device,
            bits_per_weight=quant_metadata["bits_per_weight"],
            runs=runs,
            warmup=warmup,
            input_dtype=input_dtype,
        )

        # unpack absolute results into dictionary
        (
            pytorch_metric_dict["quantized_model_size"],
            pytorch_metric_dict["quantized_median_latency"],
            pytorch_metric_dict["quantized_p95_latency"],
            pytorch_metric_dict["quantized_p99_latency"],
        ) = quantized_pytorch_size_latency

        # get and unpack results relative to baseline
        (
            pytorch_metric_dict["relative_model_size"],
            pytorch_metric_dict["relative_median_latency"],
            pytorch_metric_dict["relative_p95_latency"],
            pytorch_metric_dict["relative_p99_latency"],
        ) = assess_relative_performance(
            quantized_model_performance=quantized_pytorch_size_latency,
            base_model_performance=baseline_size_latency_results["pytorch"],
        )

        output_dict[config_name] = QuantizationResult(
            config=quant_metadata, pytorch_result=pytorch_metric_dict
        )

        if weight_only:
            if print_debug:
                print(f"Evaluating {config_name} on ONNX and PT2")

            onnx_metric_dict: Dict[str, Any] = {}
            pt2_metric_dict: Dict[str, Any] = {}

            # get the results
            try:
                quantized_onnx_size_latency = evaluate_onnx_latency_and_size(
                    model=quant_model,
                    sample_input=sample_input,
                    device=evaluation_device,
                    runs=runs,
                    warmup=warmup,
                    input_dtype=input_dtype,
                )
                (
                    onnx_metric_dict["quantized_model_size"],
                    onnx_metric_dict["quantized_median_latency"],
                    onnx_metric_dict["quantized_p95_latency"],
                    onnx_metric_dict["quantized_p99_latency"],
                ) = quantized_onnx_size_latency
                (
                    onnx_metric_dict["relative_model_size"],
                    onnx_metric_dict["relative_median_latency"],
                    onnx_metric_dict["relative_p95_latency"],
                    onnx_metric_dict["relative_p99_latency"],
                ) = assess_relative_performance(
                    quantized_model_performance=quantized_onnx_size_latency,
                    base_model_performance=baseline_size_latency_results["onnx"],
                )

                output_dict[config_name]["onnx_result"] = onnx_metric_dict
            except Exception as e:
                print(f"Skipping {config_name} ONNX evaluation: {e}")

            try:
                quantized_pt2_size_latency = evaluate_pt2_latency_and_size(
                    model=quant_model,
                    sample_input=sample_input,
                    device=evaluation_device,
                    runs=runs,
                    warmup=warmup,
                    input_dtype=input_dtype,
                )

                (
                    pt2_metric_dict["quantized_model_size"],
                    pt2_metric_dict["quantized_median_latency"],
                    pt2_metric_dict["quantized_p95_latency"],
                    pt2_metric_dict["quantized_p99_latency"],
                ) = quantized_pt2_size_latency

                (
                    pt2_metric_dict["relative_model_size"],
                    pt2_metric_dict["relative_median_latency"],
                    pt2_metric_dict["relative_p95_latency"],
                    pt2_metric_dict["relative_p99_latency"],
                ) = assess_relative_performance(
                    quantized_model_performance=quantized_pt2_size_latency,
                    base_model_performance=baseline_size_latency_results["pt2"],
                )

                output_dict[config_name]["pt2_result"] = pt2_metric_dict

            except Exception as e:
                print(f"Skipping {config_name} PT2 evaluation: {e}")

    if print_debug:
        print("Actual Evaluated Configs")
        print(output_dict.keys())

    return output_dict


if __name__ == "__main__":
    from data.load_data import get_superconductivity_data

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
        test_fraction=0.2, random_seed=12, n_workers=4, batch_n=128
    )

    print("running ptq, full")
    train_loader_full_output = run_ptq(
        base_model=test_model,
        dataloader=train_loader,
        evaluation_device=device,
        batch_size=128,
        print_debug=True,
        weight_only=False,
    )
    print("Result")
    train_loader_full_df = ptq_results_to_dataframe(train_loader_full_output)
    print(train_loader_full_df.head())
    print(train_loader_full_df.info())
    train_loader_full_df.to_csv(
        Path.cwd() / "data" / "output" / f"baseline_model_results_{device}.csv"
    )
    
    # testing the size estimation
    train_loader_relative_size_df = train_loader_full_df.query("base_metric == 'model_size' and relative == True")
    print(train_loader_relative_size_df.head(n=30))

    print("running ptq, weight only")
    train_loader_weight_output = run_ptq(
        base_model=test_model,
        dataloader=train_loader,
        evaluation_device=device,
        batch_size=128,
        print_debug=True,
        weight_only=True,
    )
    print("Result")
    train_loader_weight_df = ptq_results_to_dataframe(train_loader_weight_output)
    print(train_loader_weight_df.head())
    print(train_loader_weight_df.info())
    train_loader_weight_df.to_csv(
        Path.cwd()
        / "data"
        / "output"
        / f"baseline_model_results_weight_only_{device}.csv"
    )
