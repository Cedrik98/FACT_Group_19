"""This module runs experiments on the fine-tuned language models."""
import gc
import json
import os
import numpy as np
import typing
from argparse import ArgumentParser, Namespace
from pathlib import Path

import nltk
import textattack
import torch
from datasets.arrow_dataset import Dataset
from textattack.models.wrappers.huggingface_model_wrapper import (
    HuggingFaceModelWrapper,
)

from src.attack.attack import perform_attack
from src.attack.gen_attacker import generate_attacker
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
        args.dataset, args.seed, args.number_of_samples
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
    results = perform_attack(data, args, attacker, stopwords, str(output_path))

    ins = []
    rc = []
    abs = []
    sim = []

    for result in results:
        i = result["ins_score"]
        r = result["rc_score"]
        a = result["abs_score"]
        s = result["sim_score"]

        if i:
            ins.append(i)

        if r:
            rc.append(r)

        if a:
            abs.append(a)

        if s:
            sim.append(s)

    print("INS")
    print(np.mean(np.array(ins)))
    print(np.std(np.array(ins)))

    print("RC")
    print(np.mean(np.array(rc)))
    print(np.std(np.array(rc)))

    print("ABS")
    print(np.mean(np.array(abs)))
    print(np.std(np.array(abs)))

    print("SIM")
    print(np.mean(np.array(sim)))
    print(np.std(np.array(sim)))


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
    parser.add_argument(
        "--lime-sr",
        type=int,
        default=None,
        help="Lime sampling rate",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="The top-n feature threshold - Which features are required"
        "to be held constant between each iteration of the search",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=MODELS,
        default="bert-base-uncased",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=DATASETS,
        default="imdb",
        help="Dataset to use for experiment",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="label",
        help="Name of the column containing labels in the dataset",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default="text",
        help="Name of the column containing the input text in the dataset",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for running the models.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batchsize to use for generating datapoints on which to train"
        "LIME",
    )
    parser.add_argument(
        "--max-candidate",
        type=int,
        default=10,
        help="Max Candidates - The number of nearest neighbors calculated"
        "for word replacement",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.5,
        help="Not used in the paper's experiments, this sets a similarity"
        "threshold to allow early termination of the search once a"
        "desired dissimilarity between the original explanation and"
        "the perturbed explanation is reached.",
    )

    parser.add_argument(
        "--num",
        type=int,
        default=5,
        help="The number of documents to attack (Reccomended due to time only"
        "to choose a small subset of the included datasets)",
    )
    parser.add_argument(
        "--modify-rate",
        type=float,
        default=0.2,
        help="Modify Rate - The maximum percentage of possible perturbed words"
        "as a percentage of the document",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum length for each datasample, if it is too long, it is"
        "truncated",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=10,
        help="Minimum length for each datasample, a sample is skipped if it is"
        "shorter than this length",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12,
        help="Random seed to use",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["ga", "random", "truerandom", "xaifooler", "inherent"],
        default="random",
        help="Search method when looking for the ideal pertubation of the text"
        "for fooling LIME",
    )
    parser.add_argument(
        "--crossover",
        type=str,
        choices=["uniform", "1point"],
        default="1point",
        help="Argument that specifies the number of crossover done for the GA"
        "(genetic algorithm) search method",
    )
    parser.add_argument(
        "--parent-selection",
        type=str,
        choices=["roulette", "truncation"],
        default="truncation",
        help="Argument that specifies how parents are selected for the GA"
        "(genetic algorithm) search method",
    )
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
        help="Which measure to use for explanation"
        "comparison during the search process",
    )
    parser.add_argument(
        "--rbo-p",
        type=float,
        default=0.8,
        help="RBO-p - for the RBO similarity measure, the weight"
        "parameter that controls the top-weightedness",
    )
    parser.add_argument(
        "--number-of-samples",
        type=int,
        default=25000,
        help="Number of datapoints sampled from the dataset",
        required=False,
    )
    parser.add_argument(
        "--surrage-model",
        type=str,
        choices=["decision_tree", "logistic_regression"],
        default="logistic_regression",
        help="Surrogate model to use for training LIME",
        required=False,
    )
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--rerun", action="store_true", default=False)

    run(parser.parse_args())
