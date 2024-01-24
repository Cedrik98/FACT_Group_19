"""This module runs experiments on the fine-tuned language models."""
import gc
import json
import os
import typing
from argparse import ArgumentParser, Namespace
from pathlib import Path

import nltk
import textattack
import torch
from datasets.arrow_dataset import Dataset
from textattack.models.wrappers.huggingface_model_wrapper import HuggingFaceModelWrapper

from src.attack_recipes.attack import perform_attack
from src.attack_recipes.gen_attacker import generate_attacker
from src.constants import EXPERIMENT_LOGGING_PATH, HF_ACCOUNT
from src.dataset import DATASETS, load_data, process_experiment_data
from src.model import MODELS, load_trained_model_and_tokenizer


def run_experiment(args: Namespace):
    """Runs one experiment with given configuration."""
    # Set paths for storing information.
    output_path = Path(
        f"{EXPERIMENT_LOGGING_PATH}/{args.model}-{args.dataset}-{args.method}-{args.similarity_measure}/"
    )
    text_attack_logging_file = Path("attack_log.csv")
    text_attack_logging_path = output_path / text_attack_logging_file

    # Creates directory if it does not yet exist.
    if not output_path.exists():
        os.makedirs(output_path)

    # Dumps all passed arguments into a .json file.
    with open(output_path / "config.json", "w") as file:
        json.dump(args.__dict__, file, indent=4)

    # Load model
    model, tokenizer = load_trained_model_and_tokenizer(
        args.model, args.dataset, HF_ACCOUNT
    )

    if args.max_length:
        tokenizer.model_max_length = args.max_length  # type: ignore

    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

    # Load all necessary data
    _, dataset_test, categories = load_data(
        args.dataset, args.seed_dataset, args.number_of_samples
    )
    dataset = textattack.datasets.HuggingFaceDataset(dataset_test)
    dataset: Dataset = typing.cast(Dataset, dataset._dataset)

    if args.debug:
        dataset.shuffle()
        dataset = dataset.select(range(10))

    stopwords = set(nltk.corpus.stopwords.words("english"))  # type: ignore

    # Preprocess data
    data, categories = process_experiment_data(dataset, args, stopwords)

    # Generate attacker
    attacker = generate_attacker(
        args, model_wrapper, categories, text_attack_logging_path
    )

    print(attacker)

    # Perform attack
    results, rbos, sims = perform_attack(
        data, args, attacker, stopwords, str(output_path)
    )
    print(results)


def run(args: Namespace):
    """Entry point for running all experiments."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:24"
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    gc.collect()
    torch.cuda.empty_cache()

    run_experiment(args)


if __name__ == "__main__":
    parser = ArgumentParser(description="XAIFOOLER Experiment")
    # All of these should have a help comment!!!
    parser.add_argument("--lime-sr", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument(
        "--model",
        type=str,
        choices=MODELS,
        default="bert-base-uncased",
    )
    parser.add_argument("--dataset", type=str, choices=DATASETS, default="imdb")
    parser.add_argument("--label-col", type=str, default="label")
    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--batch-size", type=int, default=512
    )  # TODO: change back to 512
    parser.add_argument("--max-candidate", type=int, default=10)
    parser.add_argument("--success-threshold", type=float, default=0.5)
    parser.add_argument("--rbo-p", type=float, default=0.8)
    parser.add_argument("--num", type=int, default=5)
    parser.add_argument("--modify-rate", type=float, default=0.2)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--min-length", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--seed-dataset", type=int, default=12)
    parser.add_argument(
        "--method",
        type=str,
        choices=["ga", "random", "truerandom", "xaifooler", "inherent"],
        default="random",
    )
    parser.add_argument("--search-method", type=str, default="default")
    parser.add_argument(
        "--crossover", type=str, choices=["uniform", "1point"], default="1point"
    )
    parser.add_argument(
        "--parent-selection",
        type=str,
        choices=["roulette", "truncation"],
        default="truncation",
    )
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--rerun", action="store_true", default=False)
    parser.add_argument(
        "--similarity-measure",
        type=str,
        choices=[
            "rbo",
            "l2",
            "com",
            "com_rank_weighted",
            "com_proportional",
            "jaccard",
            "kendall",
            "spearman",
            "jaccard_weighted",
            "kendall_weighted",
            "spearman_weighted",
        ],
        default="rbo",
    )
    parser.add_argument(
        "--number-of-samples",
        type=int,
        default=25000,
        help="Number of datapoints sampled from the dataset",
        required=False,
    )

    run(parser.parse_args())
