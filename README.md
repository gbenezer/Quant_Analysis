# Regression Model Quantization Analysis Toolkit

A research toolkit for evaluating the impact of post-training quantization (PTQ) on regression models. The toolkit measures trade-offs between accuracy (MAE), model size, and inference latency across multiple quantization schemes and inference backends.

---

## Overview

This project explores how quantization affects MLP regression models trained on the [UCI Superconductivity dataset](https://archive.ics.uci.edu/dataset/464/superconductivty+data). It covers the full pipeline from architecture sampling and model training, through PTQ application and multi-backend evaluation, to result visualization.

**Key capabilities:**
- Design space exploration via Sobol quasi-random sampling over MLP architectures
- Post-training quantization using [TorchAO](https://github.com/pytorch/ao) (Int8, Float8 weight-only and weight-activation schemes)
- Multi-backend inference evaluation: PyTorch eager, ONNX Runtime, and PyTorch 2.0 compiled models
- Batch normalization fusion prior to quantization
- Cluster-ready multiprocessing with CUDA process isolation

---

## Repository Structure

```
Quant_Analysis/
├── pyproject.toml
├── data/
│   ├── load_data.py                   # Dataset classes and DataLoader factories
│   └── output/                        # Custom experiment (from script files) output data
│       ├── csv/                       # Experiment result CSVs
│       ├── logs/                      # Raw and filtered experiment logs
│       └── figures/                   # Custom script generated Plotly HTML visualizations
├── src/quant_analysis/
│   ├── model_architecture/
│   │   ├── simple_mlp.py              # SimpleMLP PyTorch module
│   │   ├── model_configs.py           # SimpleMLPConfig dataclass + save/load helpers
│   │   ├── simple_mlp_sampler.py      # Sobol-based architecture sampling
│   │   └── superconductor_mlp_lightning.py  # Lightning training wrapper
│   ├── quantization/ptq/
│   │   ├── run_ptq.py                 # Main PTQ evaluation pipeline
│   │   ├── quantize_ptq.py            # TorchAO quantization application + BN fusion
│   │   └── ptq_config_metadata.py     # Quantization scheme registry
│   ├── metric_calculation/
│   │   ├── evaluate_mean_absolute_error.py  # MAE computation
│   │   └── evaluate_size_and_latency.py     # Size estimation and latency benchmarking
│   ├── model_export/
│   │   ├── export_mlp_to_onnx.py      # ONNX export
│   │   └── export_mlp_to_pt2.py       # PyTorch 2.0 export
│   ├── model_loading/
│   │   └── load_mlp_from_pth.py       # Load SimpleMLP from .pth state dicts
│   └── data_processing/
│       └── ptq_result_to_dataframe.py # Convert result dicts to pandas DataFrames
├── scripts/
│   ├── design_space_sampling_experiment.py             # Design space experiment (local GPU, 32 configs, single eval run)
│   ├── cluster_design_space_sampling_experiment.py     # Design space experiment variant for cluster V100 (64 configs, 3 eval runs)
│   ├── model_cpu_experiment.py                         # Baseline model variance experiment on local CPU
│   ├── model_local_gpu_experiment.py                   # Baseline model variance experiment on local GPU
│   ├── model_cluster_experiment.py                     # Baseline model variance experiment on cluster H200 GPU
│   ├── design_space_experiment_data_processing.py      # Merges multi-run cluster CSVs into consolidated files
│   ├── design_space_experiment_visualization.py        # Plotly visualizations from design space experiment results
│   ├── variance_experiment_visualization.py            # Plotly violin plots comparing PTQ variance across devices/locations
│   ├── local_experiment_execution_script.sh            # Local execution wrapper for baseline variance experiment
│   ├── local_design_experiment_execution_script.sh     # Local execution wrapper for design experiment (with log capture)
│   ├── cluster_experiment_execution_script.sh          # SLURM job script for baseline model variance experiment
│   └── cluster_design_experiment_execution_script.sh   # SLURM job script for design space sampling experiment
├── models/
│   ├── state_dicts/                   # Trained model weights (.pth)
│   ├── configs/                       # Architecture configs (JSON)
│   ├── checkpoints/                   # PyTorch Lightning checkpoints
│   ├── onnx/                          # Exported ONNX models
│   └── pt2/                           # Exported PT2 models
└── docs/                              # Architecture diagrams and result figures
```

---

## Installation

**Requirements:** Python 3.12+, a CUDA-capable GPU is recommended.

```bash
git clone https://github.com/gbenezer/Quant_Analysis.git
cd Quant_Analysis
pip install -e .
```

Dependencies are declared in `pyproject.toml`. Key packages include:

| Category | Packages |
|---|---|
| ML Framework | `torch >= 2.10`, `lightning >= 2.6` |
| Quantization | `torchao >= 0.15` |
| Model Export | `onnx >= 1.20`, `onnxruntime >= 1.24` |
| Data | `datasets`, `scikit-learn`, `pandas`, `ucimlrepo` |
| Visualization | `plotly` |

---

## Workflow

### 1. Design Space Sampling

`simple_mlp_sampler.py` uses [Sobol quasi-random sequences](https://en.wikipedia.org/wiki/Sobol_sequence) to efficiently sample MLP architectures. Each sample specifies the number of neurons per hidden layer, activation function (ReLU, LeakyReLU, ELU, GELU, CELU), and whether batch normalization is used.

### 2. Model Training

`superconductor_mlp_lightning.py` trains `SimpleMLP` models on the UCI Superconductivity dataset (81 input features, 1 regression target) using PyTorch Lightning with L1 (MAE) loss. Trained weights and architecture configs are saved to `models/state_dicts/` and `models/configs/`. This is mainly done through the `construct_mlp` function interface.

### 3. Post-Training Quantization

`quantize_ptq.py` applies TorchAO PTQ schemes to trained models. Before quantization, batch normalization layers can be fused into adjacent linear layers using `fuse_mlp_bn()` to improve quantization accuracy.

Supported quantization schemes (defined in `ptq_config_metadata.py`), though issues may arise due to dimensionality of your models being incompatible with default quantization granularities:

| Scheme | Type | Bits/Weight | Dynamic Calibration | Minimum CUDA Compute Capability | Compatibility Notes |
|---|---|---|---|---|---|
| `int8wo` | Weight-only | 8 | No | 8.6 | |
| `float8wo` | Weight-only | 8 | No | 8.6 | ONNX incompatible given `float8e4m3fn` default TorchAO float format |
| `float8_dynamic_activation_float8_weight` | Weight + Activation | 8 | Yes | 8.6 | |
| `float8_static_activation_float8_weight` | Weight + Activation | 8 | No | 8.9 | May silently fully or partially no-op on CPU |
| `int8_dynamic_activation_int8_weight` | Weight + Activation | 8 | Yes | 8.6 | |

### 4. Evaluation

`run_ptq.py` orchestrates the full evaluation loop:

- **Accuracy**: MAE computed over the full test set for each quantized model
- **Size**: Estimated from parameter count and bits-per-weight
- **Latency**: Benchmarked with configurable warmup and evaluation runs, reporting median, p95, and p99

Evaluation backends:
- **PyTorch eager** — direct inference with the quantized model
- **ONNX Runtime** — via export to `.onnx` (weight-only schemes)
- **PyTorch 2.0** — via `torch.export` compilation (weight-only schemes)

Results are returned as nested dictionaries and can be converted to a `pandas.DataFrame` using `ptq_result_to_dataframe.py`.

### 5. Design Space Experiment

`scripts/design_space_sampling_experiment.py` runs the full pipeline end-to-end:

1. Samples 32 MLP architectures using Sobol sequences over 3 hidden layers (widths 128–1024, 64–512, 32–256)
2. Trains each architecture for 25 epochs on the superconductivity dataset
3. Evaluates each model with all PTQ schemes (500 inference runs, 50 warmup)
4. Saves results to `data/output/csv/`

```bash
# Run locally
bash scripts/local_experiment_execution_script.sh

# Run on an HPC cluster
sbatch scripts/cluster_experiment_execution_script.sh
```

---

## Package API

The `quant_analysis` package exposes its main components through `src/quant_analysis/__init__.py`.

### Example: Running PTQ on a Trained Model

```python
import torch
from quant_analysis import (
    SimpleMLPConfig,
    construct_mlp,
    run_ptq,
    PTQ_QUANT_CONFIG_METADATA,
    ptq_results_to_dataframe
)
from data.load_data import get_superconductivity_data

# set random seed and other globals (for this script execution)
SEED = 42
WORKERS = 4
BATCH_SIZE = 128
EPOCHS = 25
RUNS = 500
WARMUP = 50
SAVE_OUTPUT = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load superconductivity data
(
    _,
    _,
    _,
    _,
    _,
    _,
    test_loader,
) = get_superconductivity_data(
    test_fraction=0.2, 
    random_seed=SEED, 
    n_workers=WORKERS, 
    batch_n=BATCH_SIZE
)

# Define an architecture and train a model
config = SimpleMLPConfig(
    input_dim=81,
    output_dim=1,
    neurons_per_layer=[512, 256, 128],
    activation="relu",
    use_batch_norm=True,
)
model = construct_mlp(config=config, seed=SEED, name="my_model", max_epochs=EPOCHS, save_output=SAVE_OUTPUT)

# Evaluate all PTQ schemes
results = run_ptq(
    base_model=model,
    dataloader=test_loader,
    evaluation_device=device,
    batch_size=BATCH_SIZE,
    runs=RUNS,
    warmup=WARMUP,
    weight_only=False,
)

# convert the nested dictionary to a DataFrame for 
# downstream processing and custom visualization
result_dataframe = ptq_results_to_dataframe(results)
```

---

## Authors

- **Gil Benezer** - [benezer.gi@northeastern.edu](mailto:benezer.gi@northeastern.edu)
- **Brent Garey** - [garey.b@northeastern.edu](mailto:garey.b@northeastern.edu)
- **Ryon Sajnovsky** - [sajnovsky.r@northeastern.edu](mailto:sajnovsky.r@northeastern.edu)

Northeastern University
