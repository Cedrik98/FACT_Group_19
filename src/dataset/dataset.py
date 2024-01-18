""" This module provides utilities to load various datasets. """
from typing import Tuple, cast

from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict

IMDB_DATASET_NAME = "imdb"
GENDER_BIAS_DATASET_NAME = "md_gender_bias"
DATASETS = [IMDB_DATASET_NAME, GENDER_BIAS_DATASET_NAME]


def load_imdb_dataset(seed: int, number_of_samples: int) -> Tuple[DatasetDict, Dataset]:
    """This function loads the IMDB dataset."""
    dataset: DatasetDict = cast(DatasetDict, load_dataset(IMDB_DATASET_NAME))
    dataset = dataset.shuffle(seed=seed).select(range(number_of_samples))
    dataset_train_valid = dataset["train"].train_test_split(
        test_size=0.1,
        stratify_by_column="label",
        shuffle=True,
        seed=seed,
    )
    dataset_test = dataset["test"]
    return dataset_train_valid, dataset_test

# dataset = load_dataset(DATASET_NAME, "convai2_inferred", split="train").filter(
        #     lambda example: example["binary_label"] in set([0, 1])
        # )
        # dataset = dataset.shuffle(seed=seed).select(range(30000))
        # dataset = dataset.rename_column("binary_label", "label")
        # categories = dataset.unique("label")
        # dataset = dataset.train_test_split(
        #     test_size=0.2, stratify_by_column="label", shuffle=True, seed=seed
        # )
        # dataset_test = dataset["test"]
        # dataset_train_valid = dataset["train"].train_test_split(
        #     test_size=0.1, stratify_by_column="label", shuffle=True, seed=seed
        # )

def load_gender_bias_dataset(seed: int, number_of_samples: int) -> Tuple[DatasetDict, Dataset]:
    """This function loads the Gender bias dataset."""
    bi_set = set([0, 1])
    dataset: DatasetDict = cast(DatasetDict, load_dataset(GENDER_BIAS_DATASET_NAME, "convai2_inferred", split="train").filter(lambda example: example["binary_label"] in bi_set))

    dataset = dataset.shuffle(seed=seed).select(range(number_of_samples))
    dataset = dataset.rename_column("binary_label", "label")
    categories = dataset.unique("label")
    dataset = dataset.train_test_split(
        test_size=0.2, stratify_by_column="label", shuffle=True, seed=seed
    )
    dataset_test = dataset["test"]
    dataset_train_valid = dataset["train"].train_test_split(
        test_size=0.1, stratify_by_column="label", shuffle=True, seed=seed
    )
    return dataset_train_valid, dataset_test


def load_data(dataset_name: str, seed: int, number_of_samples: int) -> Tuple[DatasetDict, Dataset]:
    """
    This function loads a dataset depending on the passed dataset_name
    and shuffles data using the passed seed.
    """
    print(f"Loading the {dataset_name} dataset")

    if dataset_name == IMDB_DATASET_NAME:
        return load_imdb_dataset(seed=seed, number_of_samples=number_of_samples)
    elif dataset_name == GENDER_BIAS_DATASET_NAME:
        return load_gender_bias_dataset(seed=seed, number_of_samples=number_of_samples)

    # Other datasets can be added below

    raise ValueError(f"No dataset: {dataset_name}")


def load_test_data(dataset_name: str, seed: int):
    """
    This function loads just the test data of a dataset as a HuggingFaceDataset
    """
    _, test_data = load_data(dataset_name, seed)
    # return HuggingFaceDataset(test_data)
    # Seems the do not use the utility from the HuggingFaceDataset class
    return test_data
