# script for running an experiment on the cluster with H200 GPU

from pathlib import Path

import pandas as pd
import torch
import torch.multiprocessing as mp

from data.load_data import get_superconductivity_data
from src.quant_analysis.data_processing.ptq_result_to_dataframe import (
    ptq_results_to_dataframe,
)
from src.quant_analysis.model_architecture.model_configs import SimpleMLPConfig
from src.quant_analysis.model_architecture.superconductor_mlp_lightning import (
    construct_mlp,
)
from src.quant_analysis.quantization.ptq.run_ptq import run_ptq, run_ptq_isolated

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # Define globals for script
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    OUTPUT_PATH = Path.cwd() / "data" / "output" / "csv"
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    NUMBER_TRAINING_EPOCHS = 25
    NUMBER_TRAINING_RUNS = 10
    NUMBER_EVALUATE_RUNS = 10
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
    DATALOADER_KWARGS = dict(
        test_fraction=0.2, random_seed=SEED, n_workers=4, batch_n=128
    )

    full_ptq_dataframe_list = []
    weight_only_ptq_dataframe_list = []

    for train_run in range(NUMBER_TRAINING_RUNS):
        print(f"training model in training run {train_run + 1}")
        test_model = construct_mlp(
            config=base_config,
            name=f"base_model_FP32_train_run_{train_run + 1}",
            max_epochs=NUMBER_TRAINING_EPOCHS,
            seed=SEED,
        )
        test_model.share_memory()  # required for passing model to subprocess

        for eval_run in range(NUMBER_EVALUATE_RUNS):
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
            )
            train_loader_full_df = ptq_results_to_dataframe(train_loader_full_output)
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
            )
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
            )
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
            )
            test_loader_weight_df = ptq_results_to_dataframe(test_loader_weight_output)
            test_loader_weight_df = test_loader_weight_df.assign(
                train_run=(train_run + 1), eval_run=(eval_run + 1), split="test"
            )
            weight_only_ptq_dataframe_list.append(test_loader_weight_df)

    full_ptq_dataframe = pd.concat(full_ptq_dataframe_list)
    weight_only_ptq_dataframe = pd.concat(weight_only_ptq_dataframe_list)

    full_ptq_dataframe.to_csv((OUTPUT_PATH / "full_ptq_baseline_experiment_data.csv"))
    weight_only_ptq_dataframe.to_csv(
        (OUTPUT_PATH / "weight_only_ptq_baseline_experiment_data.csv")
    )
