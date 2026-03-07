import copy
import inspect
from numbers import Real
from pathlib import Path
from typing import Any, Dict, NotRequired, Tuple, TypedDict, Union

import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_linear_bn_eval
from torch.utils.data import DataLoader
from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
    Float8DynamicActivationInt4WeightConfig,
    Float8StaticActivationFloat8WeightConfig,
    Float8WeightOnlyConfig,
    Int4Tensor,
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
    quantize_,
)

from src.quant_analysis.metric_calculation import (
    assess_relative_performance,
    evaluate_mae,
    evaluate_onnx_latency_and_size,
    evaluate_pt2_latency_and_size,
    evaluate_pytorch_latency_and_estimate_size,
)
from src.quant_analysis.model_architecture import SimpleMLP
from src.quant_analysis.quantization.ptq.ptq_config_metadata import (
    PTQ_QUANT_CONFIG_METADATA,
    PTQ_WEIGHT_ONLY_CONFIG_METADATA,
    ConfigAndMetadataPTQ,
)
from src.quant_analysis.quantization.ptq.quantize_ptq import fuse_mlp_bn, quantize_ptq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
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
            device=device,
            runs=runs,
            warmup=warmup,
        )

    # iterate through the quantized models for evaluation
    if print_debug:
        print("Evaluating quantized model performances")

    for config_name, (quant_model, quant_metadata) in model_dictionary.items():
        if print_debug:
            print(f"Evaluating {config_name} on PyTorch")

        # evaluate PyTorch performance
        pytorch_metric_dict: Dict[str, Any] = {}

        # MAE
        try:
            pytorch_metric_dict["quantized_MAE"] = evaluate_mae(
                model=quant_model, dataloader=dataloader
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
            quantized_onnx_size_latency = evaluate_onnx_latency_and_size(
                model=quant_model,
                sample_input=sample_input,
                device=evaluation_device,
                runs=runs,
                warmup=warmup,
            )
            quantized_pt2_size_latency = evaluate_pt2_latency_and_size(
                model=quant_model,
                sample_input=sample_input,
                device=evaluation_device,
                runs=runs,
                warmup=warmup,
            )

            # unpack absolute results into dictionary

            (
                onnx_metric_dict["quantized_model_size"],
                onnx_metric_dict["quantized_median_latency"],
                onnx_metric_dict["quantized_p95_latency"],
                onnx_metric_dict["quantized_p99_latency"],
            ) = quantized_onnx_size_latency
            (
                pt2_metric_dict["quantized_model_size"],
                pt2_metric_dict["quantized_median_latency"],
                pt2_metric_dict["quantized_p95_latency"],
                pt2_metric_dict["quantized_p99_latency"],
            ) = quantized_pt2_size_latency

            # get and unpack results relative to baseline
            (
                onnx_metric_dict["relative_model_size"],
                onnx_metric_dict["relative_median_latency"],
                onnx_metric_dict["relative_p95_latency"],
                onnx_metric_dict["relative_p99_latency"],
            ) = assess_relative_performance(
                quantized_model_performance=quantized_onnx_size_latency,
                base_model_performance=baseline_size_latency_results["onnx"],
            )
            (
                pt2_metric_dict["relative_model_size"],
                pt2_metric_dict["relative_median_latency"],
                pt2_metric_dict["relative_p95_latency"],
                pt2_metric_dict["relative_p99_latency"],
            ) = assess_relative_performance(
                quantized_model_performance=quantized_pt2_size_latency,
                base_model_performance=baseline_size_latency_results["pt2"],
            )

            output_dict[config_name]["onnx_result"] = onnx_metric_dict
            output_dict[config_name]["pt2_result"] = pt2_metric_dict

    return output_dict


# def run_full_ptq(
#     base_model: nn.Module,
#     dataloader: DataLoader,
#     evaluation_device: str | torch.device = "cpu",
#     batch_size: int = 128,
#     latency_measurements: int = 500,
#     warmup_inferences: int = 50,
#     print_debug: bool = False,
# ):

#     if print_debug:
#         print("constructing models")

#     # construct and quantize the models
#     model_dynamic_f8a_f8w = quantize_ptq(
#         base_model,
#         Float8DynamicActivationFloat8WeightConfig,
#         quantize_device=evaluation_device,
#     )

#     model_static_f8a_f8w = quantize_ptq(
#         base_model,
#         Float8StaticActivationFloat8WeightConfig,
#         is_static=True,
#         data=dataloader,
#         quantize_device=evaluation_device,
#     )

#     model_dynamic_i8a_i8w = quantize_ptq(
#         base_model,
#         Int8DynamicActivationInt8WeightConfig,
#         quantize_device=evaluation_device,
#     )

#     model_i8w = quantize_ptq(
#         base_model, Int8WeightOnlyConfig, quantize_device=evaluation_device, version=2
#     )
#     model_f8w = quantize_ptq(
#         base_model, Float8WeightOnlyConfig, quantize_device=evaluation_device
#     )
#     model_i4w = quantize_ptq(
#         base_model, Int4WeightOnlyConfig, quantize_device=evaluation_device
#     )

#     model_config_name_bit_list = [
#         (model_dynamic_f8a_f8w, "Float8DynamicActivationFloat8WeightConfig", 8),
#         (model_static_f8a_f8w, "Float8StaticActivationFloat8WeightConfig", 8),
#         (model_dynamic_i8a_i8w, "Int8DynamicActivationInt8WeightConfig", 8),
#         (model_i8w, "Int8WeightOnlyConfig", 8),
#         (model_f8w, "Float8WeightOnlyConfig", 8),
#         (model_i4w, "Int4WeightOnlyConfig", 4),
#     ]

#     # this configuration is a bit wonky
#     try:
#         model_f8a_i4w = quantize_ptq(
#             base_model,
#             Float8DynamicActivationInt4WeightConfig,
#             quantize_device=evaluation_device,
#         )
#         model_config_name_bit_list.append(
#             (model_f8a_i4w, "Float8DynamicActivationInt4WeightConfig", 4)
#         )
#     except Exception as e:
#         print(f"Skipping Float8DynamicActivationInt4WeightConfig: {e}")

#     if print_debug:
#         for model, _, _ in model_config_name_bit_list:
#             print(model)
#         if model_i4w is not None:
#             for name, module in model_i4w.named_modules():
#                 if hasattr(module, "weight"):
#                     w = module.weight
#                     if isinstance(w, Int4Tensor):
#                         print(name, "INT4 quantized")
#                     else:
#                         print(name, "not quantized")

#     # make sure that the base model is folded to ensure apples-to-apples comparison
#     if isinstance(base_model, SimpleMLP):
#         base_model = fuse_mlp_bn(base_model)

#     if print_debug and isinstance(base_model, SimpleMLP):
#         print("base_model post folding")
#         print(base_model)

#     # get the attributes of each config as a new dictionary
#     output_dict = {}

#     if print_debug:
#         print("evaluating baseline MAE")

#     baseline_MAE = evaluate_mae(model=base_model, dataloader=dataloader)

#     if print_debug:
#         print(f"Baseline MAE: {baseline_MAE}")

#     sample_input = next(iter(dataloader))[0][:batch_size].to(evaluation_device)

#     if print_debug:
#         print(f"sample input shape: {sample_input.shape}")
#         print("evaluating baseline latency and model size")

#     (
#         baseline_model_size,
#         baseline_median_latency,
#         baseline_p95_latency,
#         baseline_p99_latency,
#     ) = evaluate_pytorch_latency_and_estimate_size(
#         base_model,
#         sample_input,
#         bits_per_weight=32,
#         runs=latency_measurements,
#         warmup=warmup_inferences,
#         device=evaluation_device,
#     )

#     if print_debug:
#         print(f"starting model quantization")

#     for quantized_model, config_name, bits_per_weight in model_config_name_bit_list:
#         print(f"Current config: {config_name}")

#         if quantized_model is None:
#             continue

#         metric_dict = {}

#         # evaluate model error
#         if print_debug:
#             print(f"evaluating {config_name} MAE")

#         try:
#             metric_dict["quantized_MAE"] = evaluate_mae(
#                 model=quantized_model, dataloader=dataloader
#             )
#             metric_dict["relative_MAE"] = metric_dict["quantized_MAE"] / baseline_MAE
#         except Exception as e:
#             print(f"Skipping {config_name}: {e}")
#             metric_dict["quantized_MAE"] = None
#             metric_dict["relative_MAE"] = None

#         # evaluate model size and latency
#         try:
#             if print_debug:
#                 print(f"evaluating size and latency for {config_name}")
#             (
#                 metric_dict["quantized_model_size"],
#                 metric_dict["quantized_median_latency"],
#                 metric_dict["quantized_p95_latency"],
#                 metric_dict["quantized_p99_latency"],
#             ) = evaluate_pytorch_latency_and_estimate_size(
#                 quantized_model,
#                 sample_input,
#                 bits_per_weight=bits_per_weight,
#                 runs=latency_measurements,
#                 warmup=warmup_inferences,
#                 device=evaluation_device,
#             )

#             metric_dict["relative_model_size"] = (
#                 metric_dict["quantized_model_size"] / baseline_model_size
#             )
#             metric_dict["relative_median_latency"] = (
#                 metric_dict["quantized_median_latency"] / baseline_median_latency
#             )
#             metric_dict["relative_p95_latency"] = (
#                 metric_dict["quantized_p95_latency"] / baseline_p95_latency
#             )
#             metric_dict["relative_p99_latency"] = (
#                 metric_dict["quantized_p99_latency"] / baseline_p99_latency
#             )

#             output_dict[config_name] = copy.deepcopy(
#                 config_property_mapping[config_name]
#             )
#             output_dict[config_name].update(metric_dict)

#         except Exception as e:
#             print(f"Skipping export for {config_name}: {e}")
#             continue

#     return output_dict


# def run_weight_only_ptq(
#     base_model: nn.Module,
#     dataloader: DataLoader,
#     evaluation_device: str | torch.device = "cpu",
#     batch_size: int = 128,
#     latency_measurements: int = 500,
#     warmup_inferences: int = 50,
#     print_debug: bool = False,
# ):

#     if print_debug:
#         print("constructing models")

#     # construct and quantize the models
#     model_i8w = quantize_ptq(
#         base_model, Int8WeightOnlyConfig, quantize_device=evaluation_device, version=2
#     )
#     model_f8w = quantize_ptq(
#         base_model, Float8WeightOnlyConfig, quantize_device=evaluation_device
#     )
#     model_i4w = quantize_ptq(
#         base_model, Int4WeightOnlyConfig, quantize_device=evaluation_device
#     )

#     model_config_name_bit_list = [
#         (model_i8w, "Int8WeightOnlyConfig", 8),
#         (model_f8w, "Float8WeightOnlyConfig", 8),
#         (model_i4w, "Int4WeightOnlyConfig", 4),
#     ]

#     # make sure that the base model is folded to ensure apples-to-apples comparison
#     if isinstance(base_model, SimpleMLP):
#         base_model = fuse_mlp_bn(base_model)

#     # get the attributes of each config as a new dictionary
#     pytorch_output_dict = {}
#     onnx_output_dict = {}
#     pt2_output_dict = {}

#     if print_debug:
#         print("evaluating baseline MAE")

#     baseline_MAE = evaluate_mae(model=base_model, dataloader=dataloader)

#     if print_debug:
#         print(f"Baseline MAE: {baseline_MAE}")

#     input_dim = next(iter(dataloader))[0].shape[1]
#     sample_input = next(iter(dataloader))[0][:batch_size].to(evaluation_device)

#     if print_debug:
#         print(f"sample input shape: {sample_input.shape}")
#         print("evaluating baseline latency and model size")

#     pytorch_size_latency_results = evaluate_pytorch_latency_and_estimate_size(
#         base_model,
#         sample_input,
#         bits_per_weight=32,
#         runs=latency_measurements,
#         warmup=warmup_inferences,
#         device=evaluation_device,
#     )

#     onnx_size_latency_results = evaluate_onnx_latency_and_size(
#         base_model,
#         input_dim,
#         latency_measurements=latency_measurements,
#         warmup_inferences=warmup_inferences,
#         device=evaluation_device,
#     )

#     pt2_size_latency_results = evaluate_pt2_latency_and_size(
#         base_model,
#         input_dim,
#         num_runs=latency_measurements,
#         warmup_runs=warmup_inferences,
#         device=evaluation_device,
#     )

#     if print_debug:
#         print(f"starting model quantization")

#     for quantized_model, config_name, bits_per_weight in model_config_name_bit_list:
#         print(f"Current config: {config_name}")

#         if quantized_model is None:
#             continue

#         pytorch_metric_dict = {}
#         onnx_metric_dict = {}
#         pt2_metric_dict = {}

#         # pytorch
#         # evaluate model error
#         if print_debug:
#             print(f"evaluating {config_name} MAE")

#         pytorch_metric_dict["quantized_MAE"] = evaluate_mae(
#             model=quantized_model, dataloader=dataloader
#         )
#         pytorch_metric_dict["relative_MAE"] = (
#             pytorch_metric_dict["quantized_MAE"] / baseline_MAE
#         )

#         # evaluate model size and latency
#         try:
#             if print_debug:
#                 print(f"evaluating size and latency for {config_name}")
#             (
#                 pytorch_metric_dict["quantized_model_size"],
#                 pytorch_metric_dict["quantized_median_latency"],
#                 pytorch_metric_dict["quantized_p95_latency"],
#                 pytorch_metric_dict["quantized_p99_latency"],
#             ) = evaluate_pytorch_latency_and_estimate_size(
#                 quantized_model,
#                 sample_input,
#                 bits_per_weight=bits_per_weight,
#                 runs=latency_measurements,
#                 warmup=warmup_inferences,
#                 device=evaluation_device,
#             )

#             pytorch_metric_dict["relative_model_size"] = (
#                 pytorch_metric_dict["quantized_model_size"]
#                 / pytorch_size_latency_results[0]
#             )
#             pytorch_metric_dict["relative_median_latency"] = (
#                 pytorch_metric_dict["quantized_median_latency"]
#                 / pytorch_size_latency_results[1]
#             )
#             pytorch_metric_dict["relative_p95_latency"] = (
#                 pytorch_metric_dict["quantized_p95_latency"]
#                 / pytorch_size_latency_results[2]
#             )
#             pytorch_metric_dict["relative_p99_latency"] = (
#                 pytorch_metric_dict["quantized_p99_latency"]
#                 / pytorch_size_latency_results[3]
#             )

#             pytorch_output_dict[config_name] = copy.deepcopy(
#                 config_property_mapping[config_name]
#             )
#             pytorch_output_dict[config_name].update(pytorch_metric_dict)

#         except Exception as e:
#             print(f"Skipping export for {config_name}: {e}")
#             continue

#         # onnx
#         # evaluate model size and latency
#         try:
#             if print_debug:
#                 print(f"evaluating size and latency for {config_name}, ONNX")
#             (
#                 onnx_metric_dict["quantized_model_size"],
#                 onnx_metric_dict["quantized_median_latency"],
#                 onnx_metric_dict["quantized_p95_latency"],
#                 onnx_metric_dict["quantized_p99_latency"],
#             ) = evaluate_onnx_latency_and_size(
#                 quantized_model,
#                 input_dim=input_dim,
#                 latency_measurements=latency_measurements,
#                 warmup_inferences=warmup_inferences,
#                 device=evaluation_device,
#             )

#             onnx_metric_dict["relative_model_size"] = (
#                 onnx_metric_dict["quantized_model_size"] / onnx_size_latency_results[0]
#             )
#             onnx_metric_dict["relative_median_latency"] = (
#                 onnx_metric_dict["quantized_median_latency"]
#                 / onnx_size_latency_results[1]
#             )
#             onnx_metric_dict["relative_p95_latency"] = (
#                 onnx_metric_dict["quantized_p95_latency"] / onnx_size_latency_results[2]
#             )
#             onnx_metric_dict["relative_p99_latency"] = (
#                 onnx_metric_dict["quantized_p99_latency"] / onnx_size_latency_results[3]
#             )

#             onnx_output_dict[config_name] = copy.deepcopy(
#                 config_property_mapping[config_name]
#             )
#             onnx_output_dict[config_name].update(onnx_metric_dict)

#         except Exception as e:
#             print(f"Skipping export for {config_name}: {e}")
#             continue

#         # pt2
#         # evaluate model size and latency
#         try:
#             if print_debug:
#                 print(f"evaluating size and latency for {config_name}, PT2")
#             (
#                 pt2_metric_dict["quantized_model_size"],
#                 pt2_metric_dict["quantized_median_latency"],
#                 pt2_metric_dict["quantized_p95_latency"],
#                 pt2_metric_dict["quantized_p99_latency"],
#             ) = evaluate_pt2_latency_and_size(
#                 quantized_model,
#                 input_dim=input_dim,
#                 num_runs=latency_measurements,
#                 warmup_runs=warmup_inferences,
#                 device=evaluation_device,
#             )

#             pt2_metric_dict["relative_model_size"] = (
#                 pt2_metric_dict["quantized_model_size"] / pt2_size_latency_results[0]
#             )
#             pt2_metric_dict["relative_median_latency"] = (
#                 pt2_metric_dict["quantized_median_latency"]
#                 / pt2_size_latency_results[1]
#             )
#             pt2_metric_dict["relative_p95_latency"] = (
#                 pt2_metric_dict["quantized_p95_latency"] / pt2_size_latency_results[2]
#             )
#             pt2_metric_dict["relative_p99_latency"] = (
#                 pt2_metric_dict["quantized_p99_latency"] / pt2_size_latency_results[3]
#             )

#             pt2_output_dict[config_name] = copy.deepcopy(
#                 config_property_mapping[config_name]
#             )
#             pt2_output_dict[config_name].update(pt2_metric_dict)

#         except Exception as e:
#             print(f"Skipping export for {config_name}: {e}")
#             continue

#     return {
#         "pytorch": pytorch_output_dict,
#         "onnx": onnx_output_dict,
#         "pt2": pt2_output_dict,
#     }


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
    print(train_loader_full_output)

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
    print(train_loader_weight_output)
