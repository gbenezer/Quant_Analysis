import tempfile
import time
from pathlib import Path
from typing import Literal

import numpy as np
import onnxruntime as ort

from src.quant_analysis.model_export.export_mlp_to_onnx import export_mlp_to_onnx


def measure_latency(
    session: ort.InferenceSession, input_name, x, runs: int = 500, warmup: int = 50
):

    # warmup (ignore timing)
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
    model_path: Path,
    file_type: Literal["checkpoint", "state_dict"],
    latency_measurements: int = 500,
    warmup_inferences: int = 50,
):

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "temp_model.onnx"

        export_mlp_to_onnx(
            file_path=model_path,
            file_type=file_type,
            onnx_path=onnx_path,
        )

        # model size
        model_size_bytes = onnx_path.stat().st_size

        # ONNX Runtime session
        session = ort.InferenceSession(str(onnx_path))

        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape[1]

        x = np.random.randn(1, input_shape).astype(np.float32)

        median_latency, p95_latency, p99_latency = measure_latency(
            session, input_name, x, latency_measurements, warmup_inferences
        )

        return model_size_bytes, median_latency, p95_latency, p99_latency
