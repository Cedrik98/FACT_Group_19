""" This module provides utilities to load various datasets. """
from typing import Tuple, cast

from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from textattack.datasets import HuggingFaceDataset

IMDB_DATASET_NAME = "imdb"


def load_imdb_dataset(seed: int) -> Tuple[DatasetDict, Dataset]:
    """ This function loads the IMDB dataset. """
    dataset: DatasetDict = cast(DatasetDict, load_dataset(IMDB_DATASET_NAME))
    dataset_train_valid = dataset["train"].train_test_split(
        test_size=0.1,
        stratify_by_column="label",
        shuffle=True,
        seed=seed,
    )
    dataset_test = dataset["test"]
    return dataset_train_valid, dataset_test


def load_data(dataset_name: str, seed: int) -> Tuple[DatasetDict, Dataset]:
    """
    This function loads a dataset depending on the passed dataset_name
    and shuffles data using the passed seed.
    """
    print(f"Loading the {dataset_name} dataset")

    if dataset_name == IMDB_DATASET_NAME:
        return load_imdb_dataset(seed=seed)

    # Other datasets can be added below

    raise ValueError(f"No dataset: {dataset_name}")


def load_test_data(dataset_name: str, seed: int):
    """
    This function loads just the test data of a dataset as a HuggingFaceDataset
    """
    _, test_data = load_data(dataset_name, seed)
    return HuggingFaceDataset(test_data)
