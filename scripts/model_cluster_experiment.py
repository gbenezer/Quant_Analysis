# script for running an experiment on the cluster with H200 GPU

from pathlib import Path
import torch
from src.quant_analysis.data_processing.ptq_result_to_dataframe import (
    ptq_results_to_dataframe,
)
from src.quant_analysis.model_loading import load_mlp_from_pth
from src.quant_analysis.quantization.ptq.run_ptq import run_ptq
from data.load_data import get_superconductivity_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
    valid_loader,
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