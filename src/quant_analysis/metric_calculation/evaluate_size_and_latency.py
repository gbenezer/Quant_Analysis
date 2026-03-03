import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn


def measure_latency(
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
    model_export_dtype: torch.dtype = torch.float32,
    input_dim: int = 81,
    latency_measurements: int = 500,
    warmup_inferences: int = 50,
):

    model.eval()

    input_sample = torch.randn((1, input_dim), dtype=model_export_dtype)

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "temp_model.onnx"

        torch.onnx.export(
            model,
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

        median_latency, p95_latency, p99_latency = measure_latency(
            session,
            input_name,
            x,
            latency_measurements,
            warmup_inferences,
        )

        return model_size_bytes, median_latency, p95_latency, p99_latency
