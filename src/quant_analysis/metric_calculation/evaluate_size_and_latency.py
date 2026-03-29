import copy
import os
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn


# ================
# Helper Functions
# ================

def measure_latency_onnx(
    session: ort.InferenceSession,
    input_name: str,
    x: np.ndarray,
    runs: int = 500,
    warmup: int = 50,
):
    """
    Measure inference latency for an ONNX model.

    Runs a number of warmup iterations to stabilize performance, then
    measures execution time over multiple runs and returns latency statistics.

    Params:
        session (ort.InferenceSession): ONNX Runtime session used for inference.
        input_name (str): Name of the model input.
        x (np.ndarray): Input data for inference.
        runs (int, optional): Number of measured runs.
        warmup (int, optional): Number of warmup runs before measurement.

    Returns:
        Tuple[float, float, float]: Median, 95th percentile, and 99th percentile latencies.
    """

    # warmup
    for _ in range(warmup):
        session.run(None, {input_name: x})

    latencies = []

    # Measure inference time across multiple runs.
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, {input_name: x})
        latencies.append(time.perf_counter() - start)

    latencies = np.array(latencies)

    # Compute latency statistics.
    median = float(np.median(latencies))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))

    return median, p95, p99


# size estimation compatible with all PTQ configurations
def estimate_quantized_size(model: nn.Module, bits_per_weight: int):
    """
    Estimate model size based on quantization.

    Params:
        model (nn.Module): PyTorch model.
        bits_per_weight (int): Number of bits used per weight.

    Returns:
        float: Estimated model size in bytes.
    """

    # Count total number of parameters in the model.
    total_weights = sum(p.numel() for p in model.parameters())

    # Convert bits to bytes.
    return total_weights * bits_per_weight / 8


# function to assess relative performance on latency and size
def assess_relative_performance(
    quantized_model_performance: Tuple[Union[int, float], float, float, float],
    base_model_performance: Tuple[Union[int, float], float, float, float],
) -> Tuple[float, float, float, float]:
    """
    Compare quantized model performance relative to a baseline model.

    Params: 
        quantized_model_performance (Tuple): (size, median_latency, p95_latency, p99_latency) for quantized model.
        base_model_performance (Tuple): Same metrics for baseline model.

    Returns:
        Tuple[float, float, float, float]: Relative size and latency metrics..
    """

    # Convert sizes to float
    quantized_model_size_float = float(quantized_model_performance[0])
    base_model_size_float = float(base_model_performance[0])

    # Ratio < 1 means smaller / faster than baseline
    relative_model_size = quantized_model_size_float / base_model_size_float

    # -- Latency ratios (lower is better) --
    # median latency
    relative_median_latency = quantized_model_performance[1] / base_model_performance[1]
    # p95 latency
    relative_p95_latency = quantized_model_performance[2] / base_model_performance[2]
    # p99 latency
    relative_p99_latency = quantized_model_performance[3] / base_model_performance[3]

    return (
        relative_model_size,
        relative_median_latency,
        relative_p95_latency,
        relative_p99_latency,
    )


# evaluation functions
def evaluate_onnx_latency_and_size(
    model: nn.Module,
    sample_input: torch.Tensor,
    device: str | torch.device,
    runs: int = 200,
    warmup: int = 50,
    input_dtype: torch.dtype = torch.float32,
):
    """
    Export a PyTorch model to ONNX and evaluate its latency and size.

    Params:
        model (nn.Module): PyTorch model to evaluate.
        sample_input (torch.Tensor): Example input used for export and inference.
        device (str or torch.device): Device to run evaluation on.
        runs (int, optional): Number of inference runs.
        warmup (int, optional): Number of warmup runs. 
        input_dtype (torch.dtype, optional):Data type for inputs. 

    Returns:
        Tuple[int, float, float, float]: Model size in bytes, median latency, p95 latency, p99 latency.
    """

    # Work on a copy to avoid mutating original model.
    model = copy.deepcopy(model)
    model.eval()

    # Temporary directory.
    with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
        onnx_path = Path(tmpdir) / "temp_model.onnx"

        # Ensure export happens on same device as model parameters.
        export_device = next(model.parameters()).device
        input_for_export = sample_input.to(device=export_device, dtype=input_dtype)

        # Export PyTorch model to ONNX format.
        torch.onnx.export(
            model,
            (input_for_export,),
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=18,
            dynamo=True,
        )

        # Get serialized ONNX file size.
        model_size_bytes = onnx_path.stat().st_size

        # Configure ONNX Runtime threading.
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = int(os.environ.get("ORT_NUM_THREADS", 4))
        sess_options.inter_op_num_threads = int(os.environ.get("ORT_NUM_THREADS", 4))

        # Create inference session.
        session = ort.InferenceSession(str(onnx_path), sess_options=sess_options)
        input_name = session.get_inputs()[0].name
        x = input_for_export.cpu().numpy()

        # Measure latency.
        median_latency, p95_latency, p99_latency = measure_latency_onnx(
            session,
            input_name,
            x,
            runs,
            warmup,
        )

        return model_size_bytes, median_latency, p95_latency, p99_latency


def evaluate_pt2_latency_and_size(
    model: nn.Module,
    sample_input: torch.Tensor,
    device: str | torch.device,
    runs: int = 200,
    warmup: int = 50,
    input_dtype: torch.dtype = torch.float32,
) -> Tuple[int, float, float, float]:
    """
    Evaluate latency and size using PyTorch 2 export format.

    Returns:
        model size (serialized) and latency statistics.
    """

    model.eval()

    actual_device = next(model.parameters()).device
    is_cuda = actual_device.type == "cuda"
    sample_input = sample_input.to(device=actual_device, dtype=input_dtype)

    # Export model using PyTorch 2 export API.
    exported = torch.export.export(model, (sample_input,))
    module = exported.module()

    # Serialize exported model to disk to measure size.
    with tempfile.NamedTemporaryFile(suffix=".pt2", delete=False, dir="/tmp") as f:
        path = f.name

    torch.export.save(exported, path)

    model_size_bytes = os.path.getsize(path)

    # Clean up temp file.
    os.remove(path)

    # Warmup runs.
    for _ in range(warmup):
        module(sample_input)
    if is_cuda:
        torch.cuda.synchronize()

    latencies = []

    # Measure latency.
    for _ in range(runs):
        if is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        module(sample_input)
        if is_cuda:
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start)

    latencies = np.array(latencies)

    median_latency = float(np.median(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    p99_latency = float(np.percentile(latencies, 99))

    return (
        model_size_bytes,
        median_latency,
        p95_latency,
        p99_latency,
    )


def evaluate_pytorch_latency_and_estimate_size(
    model: nn.Module,
    sample_input: torch.Tensor,
    device: str | torch.device,
    bits_per_weight: int,
    runs: int = 200,
    warmup: int = 50,
    input_dtype: torch.dtype = torch.float32,
):
    """
    Export a PyTorch model to ONNX and evaluate its latency and size.

    Params:
        model (nn.Module): PyTorch model to evaluate.
        sample_input (torch.Tensor): Example input used for export and inference.
        device (str or torch.device): Device to run evaluation on.
        runs (int, optional): Number of inference runs. Default is 200.
        warmup (int, optional): Number of warmup runs. Default is 50.
        input_dtype (torch.dtype, optional): Data type for inputs. Default is float32.

    Returns:
        Tuple[int, float, float, float]: Model size in bytes, median latency, p95 latency, p99 latency.
    """

    # Try to avoid modifying original model.
    try:
        model = copy.deepcopy(model).to(device)
    except Exception:
        model = model.to(device)  # fall back to in-place move without copy

    sample_input = sample_input.to(device, dtype=input_dtype)

    model.eval()

    # Estimate size based on quantization bits.
    model_size = estimate_quantized_size(model, bits_per_weight)

    device = torch.device(device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Warmup runs.
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies = []

    # Measure latency.
    with torch.no_grad():
        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            _ = model(sample_input)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()

            latencies.append(end - start)

    latencies = np.array(latencies)

    results = (
        model_size,
        float(np.median(latencies)),
        float(np.percentile(latencies, 95)),
        float(np.percentile(latencies, 99)),
    )

    return results
