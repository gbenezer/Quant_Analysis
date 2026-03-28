# script for running an experiment on a local GPU
from pathlib import Path

import pandas as pd
import torch
import torch.multiprocessing as mp

from src.quant_analysis.data_processing.ptq_result_to_dataframe import (
    ptq_results_to_dataframe,
)
from src.quant_analysis.model_architecture.model_configs import SimpleMLPConfig
from src.quant_analysis.model_architecture.superconductor_mlp_lightning import (
    construct_mlp,
)
from src.quant_analysis.quantization.ptq.run_ptq import run_ptq_isolated

# initialize global variables
consecutive_failures = 0
full_ptq_dataframe_list = []
weight_only_ptq_dataframe_list = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
local_abort = False

EXPERIMENT_DEVICE = "local_GPU_EXP_2"
OUTPUT_PATH = Path.cwd() / "data" / "output" / "csv"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

MAX_FAILURES = 100
NUMBER_TRAINING_EPOCHS = 25
NUMBER_TRAINING_RUNS = 10
NUMBER_EVALUATE_RUNS = 10
TIMEOUT = 1200
RUNS = 500
WARMUP = 50


base_config = SimpleMLPConfig(
    input_dim=81,
    output_dim=1,
    neurons_per_layer=[512, 256, 128],
    activation="gelu",
    use_batch_norm=True,
)
# Intentional to evaluate variance
SEED = None

# dictionary to properly feed the
DATALOADER_KWARGS = dict(test_fraction=0.2, random_seed=SEED, n_workers=4, batch_n=128)


def write_csvs():
    if full_ptq_dataframe_list:
        full_ptq_dataframe = pd.concat(full_ptq_dataframe_list)
        full_ptq_dataframe.to_csv(
            OUTPUT_PATH / f"full_ptq_baseline_experiment_data_{EXPERIMENT_DEVICE}.csv"
        )
        print("Wrote full PTQ CSV.", flush=True)
    else:
        print("No full PTQ results collected.", flush=True)

    if weight_only_ptq_dataframe_list:
        weight_only_ptq_dataframe = pd.concat(weight_only_ptq_dataframe_list)
        weight_only_ptq_dataframe.to_csv(
            OUTPUT_PATH
            / f"weight_only_ptq_baseline_experiment_data_{EXPERIMENT_DEVICE}.csv"
        )
        print("Wrote weight-only PTQ CSV.", flush=True)
    else:
        print("No weight-only PTQ results collected.", flush=True)


def cancel_local(reason: str):
    global local_abort
    write_csvs()
    print(f"Aborting: {reason}", flush=True)
    local_abort = True


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    for train_run in range(NUMBER_TRAINING_RUNS):
        if local_abort:
            break
        print(f"training model in training run {train_run + 1}")
        test_model = construct_mlp(
            config=base_config,
            name=f"base_model_FP32_train_run_{train_run + 1}_{EXPERIMENT_DEVICE}",
            max_epochs=NUMBER_TRAINING_EPOCHS,
            seed=SEED,
        )
        test_model.share_memory()  # required for passing model to subprocess

        for eval_run in range(NUMBER_EVALUATE_RUNS):
            if local_abort:
                break
            eval_run_failed = False
            print(
                f"getting evaluation data for training run {train_run + 1}, evaluation run {eval_run + 1}"
            )

            print(
                "running ptq with size estimation on all configurations, train dataset"
            )
            train_loader_full_output = run_ptq_isolated(
                base_model=test_model,
                dataloader_kwargs=DATALOADER_KWARGS,
                evaluation_device=str(device),  # spawn can't pickle torch.device
                batch_size=128,
                weight_only=False,
                split="train",
                timeout=TIMEOUT,
                runs=RUNS,
                warmup=WARMUP,
            )
            if not train_loader_full_output:
                eval_run_failed = True
            else:
                train_loader_full_df = ptq_results_to_dataframe(
                    train_loader_full_output
                )
                train_loader_full_df = train_loader_full_df.assign(
                    train_run=(train_run + 1), eval_run=(eval_run + 1), split="train"
                )
                full_ptq_dataframe_list.append(train_loader_full_df)

            print(
                "running ptq with size estimation on all configurations, test dataset"
            )
            test_loader_full_output = run_ptq_isolated(
                base_model=test_model,
                dataloader_kwargs=DATALOADER_KWARGS,
                evaluation_device=str(device),
                batch_size=128,
                weight_only=False,
                split="test",
                timeout=TIMEOUT,
                runs=RUNS,
                warmup=WARMUP,
            )
            if not test_loader_full_output:
                eval_run_failed = True
            else:
                test_loader_full_df = ptq_results_to_dataframe(test_loader_full_output)
                test_loader_full_df = test_loader_full_df.assign(
                    train_run=(train_run + 1), eval_run=(eval_run + 1), split="test"
                )
                full_ptq_dataframe_list.append(test_loader_full_df)

            print(
                "running ptq, weight only configurations, actual size measurement, train set"
            )
            train_loader_weight_output = run_ptq_isolated(
                base_model=test_model,
                dataloader_kwargs=DATALOADER_KWARGS,
                evaluation_device=str(device),
                batch_size=128,
                weight_only=True,
                split="train",
                timeout=TIMEOUT,
                runs=RUNS,
                warmup=WARMUP,
            )
            if not train_loader_weight_output:
                eval_run_failed = True
            else:
                train_loader_weight_df = ptq_results_to_dataframe(
                    train_loader_weight_output
                )
                train_loader_weight_df = train_loader_weight_df.assign(
                    train_run=(train_run + 1), eval_run=(eval_run + 1), split="train"
                )
                weight_only_ptq_dataframe_list.append(train_loader_weight_df)

            print(
                "running ptq, weight only configurations, actual size measurement, test set"
            )
            test_loader_weight_output = run_ptq_isolated(
                base_model=test_model,
                dataloader_kwargs=DATALOADER_KWARGS,
                evaluation_device=str(device),
                batch_size=128,
                weight_only=True,
                split="test",
                timeout=TIMEOUT,
                runs=RUNS,
                warmup=WARMUP,
            )

            if not test_loader_weight_output:
                eval_run_failed = True
            else:
                test_loader_weight_df = ptq_results_to_dataframe(
                    test_loader_weight_output
                )
                test_loader_weight_df = test_loader_weight_df.assign(
                    train_run=(train_run + 1), eval_run=(eval_run + 1), split="test"
                )
                weight_only_ptq_dataframe_list.append(test_loader_weight_df)

            if eval_run_failed:
                consecutive_failures += 1
                if consecutive_failures >= MAX_FAILURES:
                    cancel_local("Too many consecutive worker failures")
            else:
                consecutive_failures = 0

    if full_ptq_dataframe_list:
        full_ptq_dataframe = pd.concat(full_ptq_dataframe_list)
        full_ptq_dataframe.to_csv(
            OUTPUT_PATH / f"full_ptq_baseline_experiment_data_{EXPERIMENT_DEVICE}.csv"
        )
    else:
        print("No full PTQ results collected.")

    if weight_only_ptq_dataframe_list:
        weight_only_ptq_dataframe = pd.concat(weight_only_ptq_dataframe_list)
        weight_only_ptq_dataframe.to_csv(
            OUTPUT_PATH
            / f"weight_only_ptq_baseline_experiment_data_{EXPERIMENT_DEVICE}.csv"
        )
    else:
        print("No weight-only PTQ results collected.")
