""" This module runs experiments on the fine-tuned language models."""
import gc
import os
from argparse import ArgumentParser, Namespace

import numpy
import scipy
import textattack
import torch

from src.attack_recipes.attack import perform_attack
from src.attack_recipes.gen_attacker import generate_attacker
from src.utils.data_loader import load_dataset_custom, load_stopwords, process_dataset
from src.utils.file_create import generate_filename, setup_output_environment
from src.utils.load_model import load_model_and_tokenizer

MODELS = [
    "distilbert-base-uncased-imdb-saved",
    "bert-base-uncased-imdb-saved",
    "roberta-base-imdb-saved",
    "distilbert-base-uncased-md_gender_bias-saved",
    "bert-base-uncased-md_gender_bias-saved",
    "roberta-base-md_gender_bias-saved",
    "bert-base-uncased-s2d-saved",
    "distilbert-base-uncased-s2d-saved",
    "roberta-base-s2d-saved",
]


def run_experiment(args: Namespace):
    """Runs one experiment with given configuration."""
    filename = generate_filename(args)

    setup_output_environment(filename, args)

    model, tokenizer, model_wrapper = load_model_and_tokenizer(
        args.model, args.max_length, args.device, MODELS
    )
    stopwords = load_stopwords()

    _, dataset_test, categories = load_dataset_custom(args.dataset, args.seed_dataset)
    dataset = textattack.datasets.HuggingFaceDataset(dataset_test)

    dataset = dataset._dataset

    if args.debug:
        dataset = dataset.shuffle()
        dataset = dataset.select(range(100))
        print(dataset)

    data, categories = process_dataset(dataset, args, stopwords)
    # ---------------------------------------------------
    outputName = "output"
    startIndex = 0
    csvName = outputName + str(startIndex) + "_log.csv"
    folderName = "outputName" + str(startIndex)
    # ---------------------------------------------------

    attacker = generate_attacker(args, model_wrapper, categories, csvName)

    print(attacker)

    results, rbos, sims = perform_attack(data, args, attacker, stopwords, filename)


def run(args: Namespace):
    """Entry point for running all experiments."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:24"
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    def monkeypath_itemfreq(sampler_indices):
        return zip(*numpy.unique(sampler_indices, return_counts=True))

    scipy.stats.itemfreq = monkeypath_itemfreq

    gc.collect()
    torch.cuda.empty_cache()

    run_experiment(args)


if __name__ == "__main__":
    parser = ArgumentParser(description="XAIFOOLER Experiment")
    # All of these should have a help comment!!!
    parser.add_argument("--lime-sr", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument(
        "--model", type=str, default="thaile/bert-base-uncased-imdb-saved"
    )
    parser.add_argument("--dataset", type=str, default="imdb")
    parser.add_argument("--label-col", type=str, default="label")
    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)  # TODO: change back to 512
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
        "--method", type=str, choices=["ga", "random", "truerandom"], default="random"
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
    run(parser.parse_args())
