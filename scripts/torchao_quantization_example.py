# tutorial script using torchAO for quantization

# necessary imports
# external
import copy
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import torch

# also try using the legacy torch facilities for static quantization
import torch.ao.quantization as tq
import torchao

# internal
from models import SuperconductorLightning

# globals
CHECKPOINT_PATH = Path(os.getcwd()) / "models" / "checkpoints"
PT2E_PATH = Path(os.getcwd()) / "models" / "pt2"
FILENAME = "base_model_FP32"

# first, load the model
example_base_lightning_model = SuperconductorLightning.load_from_checkpoint(
    checkpoint_path=(CHECKPOINT_PATH / (FILENAME + ".ckpt")), map_location="cpu"
)
example_base_model = example_base_lightning_model.model.eval()

# make a deep copy
model_w8a8 = copy.deepcopy(example_base_model)

# quantize the model with torchao to int8
torchao.quantization.quantize_(
    model_w8a8, config=torchao.quantization.Int8DynamicActivationInt8WeightConfig()
)

# save all the state dictionaries
torch.save(example_base_model.state_dict(), "scripts/base_model.pth")
torch.save(model_w8a8.state_dict(), "scripts/model_w8a8.pth")

# model sizes
original_size = os.path.getsize("scripts/base_model.pth")
size_8_bit = os.path.getsize("scripts/model_w8a8.pth")

print(f"Size reduction, 8 bit: {original_size / size_8_bit}")

# move all the models to the gpu
if torch.cuda.is_available():
    # move the models to the GPU
    example_base_model.to(device=torch.device("cuda")).eval()
    model_w8a8.to(device=torch.device("cuda")).eval()

    # initialize arrays for storing inference times
    base_model_inference = np.zeros(shape=(1000,))
    w8a8_model_inference = np.zeros(shape=(1000,))

    torch.cuda.synchronize()
    for _ in range(10):
        example_inputs = torch.rand(size=(128, 81), device=torch.device("cuda"))
        _ = example_base_model(example_inputs)
        _ = model_w8a8(example_inputs)

    torch.cuda.synchronize()
    for i in range(1000):
        example_inputs = torch.rand(
            size=(128, 81), device=torch.device("cuda"), dtype=torch.float32
        )
        start = time.time()
        _ = example_base_model(example_inputs)
        end = time.time()
        duration = end - start
        base_model_inference[i] = duration

    base_inference_df = pd.DataFrame(data={"inference_time": base_model_inference})
    base_inference_df = base_inference_df.assign(model="base")

    torch.cuda.synchronize()
    for i in range(1000):
        example_inputs = torch.rand(size=(128, 81), device=torch.device("cuda"))
        start = time.time()
        _ = model_w8a8(example_inputs)
        end = time.time()
        duration = end - start
        w8a8_model_inference[i] = duration

    w8a8_inference_df = pd.DataFrame(data={"inference_time": w8a8_model_inference})
    w8a8_inference_df = w8a8_inference_df.assign(model="w8a8")

    inference_df = pd.concat([base_inference_df, w8a8_inference_df], axis=0)

    figure = px.box(
        data_frame=inference_df, x="model", y="inference_time", color="model"
    )
    figure.write_html("scripts/inference_time_plots.html")
