from pathlib import Path

import numpy as np
import pandas as pd
import ramanspy as rp
import torch

path = Path.cwd() / "data" / "ILSdata.csv"
spectra, additional_features, labels = rp.datasets.adenine(file=path, download=False)
preprocessing_pipeline = rp.preprocessing.protocols.georgiev2023_P3(fingerprint=False)

print(spectra.shape)
print(spectra)
print(isinstance(spectra, pd.DataFrame))
print(spectra.columns)
print(additional_features)
print(labels)
