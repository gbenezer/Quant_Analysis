# code for a local GPU execution of an experiment (not using the multiprocessing code)
# to explore the design space
from pathlib import Path

import pandas as pd
import torch

from data.load_data import get_superconductivity_data
from src.quant_analysis.data_processing.ptq_result_to_dataframe import (
    ptq_results_to_dataframe,
)
from src.quant_analysis.model_architecture.model_configs import SimpleMLPConfig
from src.quant_analysis.model_architecture.simple_mlp_sampler import (
    generate_mlp_config_list_from_dataframe,
    generate_mlp_sample_dataframe,
)
from src.quant_analysis.model_architecture.superconductor_mlp_lightning import (
    construct_mlp,
)
from src.quant_analysis.quantization.ptq.run_ptq import run_ptq

# initialize global variables
full_ptq_dataframe_list = []
weight_only_ptq_dataframe_list = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXPERIMENT_DEVICE = "GPU"
OUTPUT_PATH = Path.cwd() / "data" / "output" / "csv"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

MAX_FAILURES = 100
NUMBER_TRAINING_EPOCHS = 25
NUMBER_EVALUATE_RUNS = 3
TIMEOUT = 1200
RUNS = 500
WARMUP = 50
NUMBER_CONFIGS = 32
BATCH_SIZE = 128

# Evaluating performance, so seed needs to be set
SEED = 32

# Construct model config dataframe, using 3 hidden layers
model_ID_list = [f"model_{i + 1}" for i in range(NUMBER_CONFIGS)]
model_IDs = pd.Series(data=model_ID_list)
model_config_dataframe = generate_mlp_sample_dataframe(
    number_samples=NUMBER_CONFIGS,
    layer_bounds=[(128, 1024), (64, 512), (32, 256)],
    test_batch_norm=False,
    random_seed=SEED,
)
model_config_dataframe["model_ID"] = model_IDs

# construct a dictionary mapping model IDs to configs
model_config_list = generate_mlp_config_list_from_dataframe(
    df=model_config_dataframe, input_dim=81, output_dim=1
)
model_configs = {
    model_key: model_config
    for (model_key, model_config) in zip(model_ID_list, model_config_list)
}

# for key, value in model_configs.items():
#     print(f"key: {key}")
#     print(f"value: {value}")

# get the test evaluation data
(
    _,
    _,
    _,
    _,
    _,
    _,
    test_loader,
) = get_superconductivity_data(
    test_fraction=0.2, random_seed=SEED, n_workers=4, batch_n=BATCH_SIZE
)

if __name__ == "__main__":
    for config_ID, config in model_configs.items():
        test_config_model = construct_mlp(
            config=config,
            seed=SEED,
            save_output=False,
            max_epochs=NUMBER_TRAINING_EPOCHS,
        )

        for eval_run in range(NUMBER_EVALUATE_RUNS):
            config_full_output_dict = run_ptq(
                base_model=test_config_model,
                dataloader=test_loader,
                evaluation_device=device,
                batch_size=BATCH_SIZE,
                runs=RUNS,
                warmup=WARMUP,
                weight_only=False,
            )

            if config_full_output_dict:
                config_full_output_df = ptq_results_to_dataframe(
                    config_full_output_dict
                )
                config_full_output_df = config_full_output_df.assign(
                    model_ID=config_ID, eval_run=(eval_run + 1), weight_only=False
                )
                full_ptq_dataframe_list.append(config_full_output_df)

            config_weight_only_output_dict = run_ptq(
                base_model=test_config_model,
                dataloader=test_loader,
                evaluation_device=device,
                batch_size=BATCH_SIZE,
                runs=RUNS,
                warmup=WARMUP,
                weight_only=True,
            )

            if config_weight_only_output_dict:
                config_weight_only_output_df = ptq_results_to_dataframe(
                    config_weight_only_output_dict
                )
                config_weight_only_output_df = config_weight_only_output_df.assign(
                    model_ID=config_ID, eval_run=(eval_run + 1), weight_only=True
                )

    if full_ptq_dataframe_list:
        full_ptq_dataframe = pd.concat(full_ptq_dataframe_list)
        full_ptq_dataframe.to_csv(
            OUTPUT_PATH
            / f"full_ptq_config_sampling_experiment_data_{EXPERIMENT_DEVICE}.csv"
        )
    else:
        print("No full PTQ results collected.")

    if weight_only_ptq_dataframe_list:
        weight_only_ptq_dataframe = pd.concat(weight_only_ptq_dataframe_list)
        weight_only_ptq_dataframe.to_csv(
            OUTPUT_PATH
            / f"weight_only_ptq_config_sampling_experiment_data_{EXPERIMENT_DEVICE}.csv"
        )
    else:
        print("No weight-only PTQ results collected.")
