import io
import os
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

def measure_latency_onnx(
    session: ort.InferenceSession,
    input_name: str,
    x: np.ndarray,
    runs: int = 500,
    warmup: int = 50,
):

    # warmup
    for _ in range(warmup):
        session.run(None, {input_name: x})

    latencies = []

    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, {input_name: x})
        latencies.append(time.perf_counter() - start)

    latencies = np.array(latencies)

    median = float(np.median(latencies))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))

    return median, p95, p99


def evaluate_onnx_latency_and_size(
    model: nn.Module,
    input_dim: int = 81,
    device: str | torch.device = "cpu",
    latency_measurements: int = 500,
    warmup_inferences: int = 50,
):

    model.eval()

    input_sample = torch.randn(
        (128, input_dim),
        dtype=torch.float32,
        device=device,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "temp_model.onnx"

        model_cpu = model.to("cpu")
        input_sample = input_sample.to("cpu")

        torch.onnx.export(
            model_cpu,
            (input_sample,),
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=18,
        )

        # model size
        model_size_bytes = onnx_path.stat().st_size

        # ONNX Runtime session
        session = ort.InferenceSession(str(onnx_path))

        input_name = session.get_inputs()[0].name
        x = input_sample.numpy()

        median_latency, p95_latency, p99_latency = measure_latency_onnx(
            session,
            input_name,
            x,
            latency_measurements,
            warmup_inferences,
        )

        return model_size_bytes, median_latency, p95_latency, p99_latency


def evaluate_pt2_latency_and_size(
    model: nn.Module,
    input_dim: int,
    num_runs: int = 200,
    warmup_runs: int = 20,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[int, float, float, float]:

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device is None
        else device
    )

    model = model.to(device)
    model.eval()

    sample_input = torch.randn(128, input_dim, device=device)

    exported = torch.export.export(model, (sample_input,))

    module = exported.module()

    with tempfile.NamedTemporaryFile(suffix=".pt2", delete=False) as f:
        path = f.name

    torch.export.save(exported, path)

    model_size_bytes = os.path.getsize(path)

    os.remove(path)

    for _ in range(warmup_runs):
        module(sample_input)

    latencies = []

    for _ in range(num_runs):
        start = time.perf_counter()
        module(sample_input)
        end = time.perf_counter()

        latencies.append(end - start)

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


def evaluate_pytorch_latency_and_size(
    model: nn.Module,
    sample_input: torch.Tensor,
    device: str | torch.device,
    runs: int = 200,
    warmup: int = 50,
):

    model = model.to(device)
    sample_input = sample_input.to(device)

    model.eval()

    model_size = sum(p.numel() * p.element_size() for p in model.parameters())

    device = torch.device(device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies = []

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
