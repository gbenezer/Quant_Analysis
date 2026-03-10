# DataSet and DataLoader logic

# re-used prior work from CS 6140 project at
# https://github.com/gbenezer/BO-for-NN-HPO-Project/blob/main/src/network/load_data.py

import numpy as np
import torch
import pandas as pd
import torch.utils.data as data
import torch.nn.functional as F
from sklearn.preprocessing import normalize, scale
from ucimlrepo import fetch_ucirepo
import ramanspy as rp
from pathlib import Path

# define Dataset class for Superconductivity data
class SuperconductivityDataset(data.Dataset):
    """
    Dataset subclass for the Superconductivity data that fetches the data from the UCI ML data repository
    """

    def __init__(
        self,
        normalize_samples=True,
        standardize_features=True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.data_object = fetch_ucirepo(id=464)
        self.features = self.data_object.data.features
        self.targets = self.data_object.data.targets
        self.feature_ndarray = self.features.to_numpy()
        if standardize_features:
            self.feature_ndarray = scale(self.feature_ndarray)
        self.target_ndarray = self.targets.to_numpy().squeeze().astype(dtype=np.float32)
        self.normalize_samples = normalize_samples
        self.metadata = self.data_object.metadata
        self.variables = self.data_object.variables
        self.number_samples = self.features.shape[0]
        self.number_features = self.features.shape[1]
        self.dtype = dtype

    def __len__(self):
        return self.number_samples

    def __getitem__(self, index):

        # get sample and target
        sample = self.feature_ndarray[index, :].reshape(1, -1)
        target = self.target_ndarray[index]

        # normalize vector if necessary
        if self.normalize_samples:
            sample = normalize(sample)

        # squeeze extra dimension out
        sample = sample.squeeze()

        # cast the two outputs to torch tensors with a given dtype
        # defaults to torch Float32
        sample = torch.tensor(sample, dtype=self.dtype)
        target = torch.tensor(target, dtype=self.dtype)

        return sample, target

# define dataset for Adenine spectral data
class AdenineSpectraDataset(data.Dataset):
    
    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        path: Path = Path.cwd() / "data" / "ILSdata.csv",
        download: bool = False,
        normalize_samples: bool = True
    ):
        super().__init__()
        
        self.spectra, self.additional_features, self.labels = rp.datasets.adenine(file=path, download=download)
        self.dtype = dtype
        self.num_samples = len(self.labels)
        self.normalize_samples = normalize_samples
        
        if isinstance(self.spectra, pd.DataFrame) and isinstance(self.labels, pd.Series):
            # store spectral axis once (shared across all samples)
            self.spectral_axis = torch.tensor(
                [float(x) for x in self.spectra.columns], dtype=self.dtype
            )
            self.num_wavenumbers = len(self.spectral_axis)
            
            # spectral intensities: (num_samples, num_wavenumbers)
            self.intensities = torch.tensor(
                self.spectra.to_numpy(), dtype=self.dtype
            )
            
            # labels
            self.label_tensor = torch.tensor(
                self.labels.to_numpy(), dtype=self.dtype
            )
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        sample = self.intensities[index]  # shape: (num_wavenumbers,)
        target = self.label_tensor[index]
        
        if self.normalize_samples:
            sample = F.normalize(sample, p=2, dim=0)  # L2 norm along spectral dim
        
        return sample, target

def get_superconductivity_data(
    test_fraction: float,
    random_seed: int,
    n_workers: int,
    batch_n: int,
    dtype: torch.dtype = torch.float32,
    validation_set: bool = False,
    valid_fraction: float = 0.1,
):
    """A factory function to retrieve and construct Datasets and Dataloaders for the superconductivity data

    Args:
        test_fraction (float): fraction of total data to use as test data
        random_seed (int): the number for seeding the random number generator
        n_workers (int): how many subprocesses the output dataloaders should use for loading data
        batch_n (int): batch size for the output dataloaders
        dtype (torch.dtype, optional): output dataloader dtype. Defaults to torch.float64.
        validation_set (bool, optional): whether or not to generate a validation set. Defaults to False.
        valid_fraction (float, optional): fraction of training data to use for validation set. Defaults to 0.1.

    Returns:
        Tuple[full_set: torch.utils.data.Dataset,
            train_set: torch.utils.data.Dataset
            valid_set: torch.utils.data.Dataset | None
            test_set: torch.utils.data.Dataset
            train_loader: torch.utils.data.DataLoader,
            valid_loader: torch.utils.data.DataLoader | None,
            test_loader: torch.utils.data.DataLoader]: datasets and dataloaders
    """
    # instantiating the full dataset
    full_dataset = SuperconductivityDataset(dtype=dtype)

    # splitting off the test dataset
    test_size = int(len(full_dataset) * test_fraction)
    non_test_size = len(full_dataset) - test_size
    seed = torch.Generator().manual_seed(random_seed)
    test_set, non_test_set = data.random_split(
        full_dataset, [test_size, non_test_size], generator=seed
    )

    # splitting the remaining set into training and validation sets
    if validation_set:
        valid_size = int(len(non_test_set) * valid_fraction)
        train_size = len(non_test_set) - valid_size
        train_set, valid_set = data.random_split(
            non_test_set, [train_size, valid_size], generator=seed
        )
        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_set,
            num_workers=n_workers,
            batch_size=batch_n,
            persistent_workers=True,
        )
    else:
        train_set = non_test_set
        valid_set = None
        valid_loader = None

    # creating the DataLoader objects
    # creating the test data DataLoader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        num_workers=n_workers,
        batch_size=batch_n,
        persistent_workers=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        num_workers=n_workers,
        batch_size=batch_n,
        persistent_workers=True,
    )

    return (
        full_dataset,
        train_set,
        valid_set,
        test_set,
        train_loader,
        valid_loader,
        test_loader,
    )
