# DataSet and DataLoader logic

# re-used prior work from CS 6140 project at
# https://github.com/gbenezer/BO-for-NN-HPO-Project/blob/main/src/network/load_data.py

import torch
import torch.utils.data as data
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import normalize, scale


# define Dataset class for Superconductivity data
class SuperconductivityDataset(data.Dataset):
    """Dataset subclass for the Superconductivity data that fetches the data from the UCI ML data repository"""

    def __init__(
        self,
        normalize_samples=True,
        standardize_features=True,
        dtype: torch.dtype = torch.float32
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
        sample  = torch.tensor(sample, dtype=self.dtype)
        target = torch.tensor(target, dtype=self.dtype)
        
        return sample, target
        

def get_superconductivity_data(
    test_fraction: float,
    random_seed: int,
    n_workers: int,
    batch_n: int,
    dtype: torch.dtype = torch.float64,
    validation_set: bool = False,
    valid_fraction: float = 0.1,
):
    """_summary_

    Args:
        test_fraction (float): fraction of total data to use as test data
        random_seed (int): the number for seeding the random number generator
        n_workers (int): how many subprocesses the output dataloaders should use for loading data
        batch_n (int): batch size for the output dataloaders
        validation_set (bool): whether or not to generate a validation set. Defaults to False.
        valid_fraction (float): fraction of training data left over to use as validation data.
            only used if validation_set = True.

    Returns:
        Dataset and DataLoader objects corresponding to training, validation, and testing sets of 
            physicochemical features of superconductors along with associated critical temperatures
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