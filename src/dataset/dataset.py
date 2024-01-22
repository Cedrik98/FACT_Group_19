""" This module provides utilities to load various datasets. """
from typing import List, Tuple, cast

from datasets import ClassLabel, concatenate_datasets, load_dataset
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict

IMDB_DATASET_NAME = "imdb"
MD_GENDER_BIAS_DATASET_NAME = "md_gender_bias"
SYMPTOM_TO_DIAGNOSIS_DATASET_NAME = "symptom_to_diagnosis"

DATASETS = [
    IMDB_DATASET_NAME,
    MD_GENDER_BIAS_DATASET_NAME,
    SYMPTOM_TO_DIAGNOSIS_DATASET_NAME,
]

IMDB_HF_DATASET_NAME = "imdb"
MD_GENDER_BIAS_HF_DATASET_NAME = "md_gender_bias"
SYMPTOM_TO_DIAGNOSIS_HF_DATASET_NAME = "gretelai/symptom_to_diagnosis"


def load_imdb_dataset(seed: int, number_of_samples: int) -> Tuple[DatasetDict, Dataset]:
    """
    This function loads the IMDB dataset that can be found here:
    https://huggingface.co/datasets/imdb
    """
    t_dataset: DatasetDict = cast(DatasetDict, load_dataset(IMDB_HF_DATASET_NAME))
    t_dataset = t_dataset.shuffle(seed=seed)
    dataset = t_dataset["train"]
    dataset = dataset.select(range(number_of_samples)[: len(dataset)])
    dataset_train_valid = dataset.train_test_split(
        test_size=0.1,
        stratify_by_column="label",
        shuffle=True,
        seed=seed,
    )
    dataset_test: Dataset = cast(Dataset, t_dataset["test"])
    return dataset_train_valid, dataset_test


def load_md_gender_bias_dataset(
    seed: int, number_of_samples: int
) -> Tuple[DatasetDict, Dataset]:
    """
    This function loads the md_gender_bias dataset that can be found here:
    https://huggingface.co/datasets/md_gender_bias
    """
    bi_set = set([0, 1])
    t_dataset: Dataset = cast(
        Dataset,
        load_dataset(
            MD_GENDER_BIAS_HF_DATASET_NAME, "convai2_inferred", split="train"
        ).filter(lambda example: example["binary_label"] in bi_set),
    )

    t_dataset = t_dataset.shuffle(seed=seed).select(
        range(number_of_samples)[: len(t_dataset)]
    )
    t_dataset = t_dataset.rename_column("binary_label", "label")
    dataset = t_dataset.train_test_split(
        test_size=0.2, stratify_by_column="label", shuffle=True, seed=seed
    )
    dataset_test = dataset["test"]
    dataset_train_valid = dataset["train"].train_test_split(
        test_size=0.1, stratify_by_column="label", shuffle=True, seed=seed
    )
    return dataset_train_valid, dataset_test


def load_symptom_to_diagnosis_dataset(seed: int, number_of_samples: int):
    """
    Loads in the symptom to diagnosis dataset that can be found here:
    https://huggingface.co/datasets/gretelai/symptom_to_diagnosis
    """
    dataset_t = cast(DatasetDict, load_dataset(SYMPTOM_TO_DIAGNOSIS_HF_DATASET_NAME))
    dataset_t1 = concatenate_datasets([dataset_t["train"], dataset_t["train"]])
    dataset_t1 = dataset_t1.shuffle(number_of_samples)
    dataset_t1 = dataset_t1.flatten_indices()
    dataset_t1 = dataset_t1.select(range(number_of_samples)[: len(dataset_t1)])
    dataset_t1 = dataset_t1.rename_column("output_text", "label")
    dataset_t1 = dataset_t1.rename_column("input_text", "text")
    dataset_t1 = dataset_t1.class_encode_column("label")
    categories = dataset_t1.unique("label")
    labels = ClassLabel(num_classes=len(categories), names=categories)
    dataset_t1 = dataset_t1.cast_column("label", labels)

    dataset = dataset_t1.train_test_split(
        test_size=0.2, stratify_by_column="label", shuffle=True, seed=seed
    )
    dataset_test = dataset["test"]
    dataset_train_valid = dataset["train"].train_test_split(
        test_size=0.1, stratify_by_column="label", shuffle=True, seed=seed
    )
    return dataset_train_valid, dataset_test


def load_data(
    dataset_name: str, seed: int, number_of_samples: int
) -> Tuple[DatasetDict, Dataset, List]:
    """
    This function loads a dataset depending on the passed dataset_name
    and shuffles data using the passed seed.
    """
    print(f"Loading the {dataset_name} dataset")

    if dataset_name == IMDB_DATASET_NAME:
        train_valid_dataset, test_dataset = load_imdb_dataset(
            seed=seed, number_of_samples=number_of_samples
        )
    elif dataset_name == MD_GENDER_BIAS_DATASET_NAME:
        train_valid_dataset, test_dataset = load_md_gender_bias_dataset(
            seed=seed, number_of_samples=number_of_samples
        )
    elif dataset_name == SYMPTOM_TO_DIAGNOSIS_DATASET_NAME:
        train_valid_dataset, test_dataset = load_symptom_to_diagnosis_dataset(
            seed=seed, number_of_samples=number_of_samples
        )
        # Other datasets can be added below
    else:
        raise ValueError(f"No dataset: {dataset_name}")

    train_dataset = train_valid_dataset["train"]
    valid_dataset = train_valid_dataset["test"]

    train_categories = train_dataset.unique("label")
    valid_categories = valid_dataset.unique("label")
    test_categories = test_dataset.unique("label")
    categories = list(set(train_categories + valid_categories + test_categories))

    return train_valid_dataset, test_dataset, categories
