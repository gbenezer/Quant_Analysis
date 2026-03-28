# code for a local GPU execution of an experiment (not using the multiprocessing code)
# to explore the design space
from pathlib import Path

import pandas as pd
import torch

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
NUMBER_TRAINING_RUNS = 1
NUMBER_EVALUATE_RUNS = 5
TIMEOUT = 1200
RUNS = 500
WARMUP = 50
NUMBER_CONFIGS = 64

# Evaluating performance, so seed needs to be set
SEED = 32

# Construct model config dataframe, using 3 hidden layers
model_IDs = pd.Series(data=[f"model_{i+1}" for i in range(NUMBER_CONFIGS)])
model_config_dataframe = generate_mlp_sample_dataframe(
    number_samples=NUMBER_CONFIGS,
    layer_bounds=[(128, 1024), (64, 512), (32, 256)],
    test_batch_norm=False,
    random_seed=SEED
)
model_config_dataframe["model_ID"] = model_IDs
model_config_list = generate_mlp_config_list_from_dataframe(df=model_config_dataframe, input_dim=81, output_dim=1)

print(model_config_dataframe.head(15))