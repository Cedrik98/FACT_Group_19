# TODO:
# - Rename this file
# - Extra functionality from run()
import random
import typing
from argparse import Namespace

import torch
from textattack import AttackArgs, Attacker
from textattack.datasets import Dataset
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.shared.attacked_text import AttackedText
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.attack_recipes import RandomBaseline
from src.dataset import load_data

ATTACK_RECIPES = {"random": RandomBaseline}


def generate_attacker(
    args: Namespace,
    model_wrapper,
    categories,
    custom_seed=None,
    greedy_search=True,
):
    """TODO: move this utility to the attackers folder"""
    attack = ATTACK_RECIPES[args.attack_recipe].build(
        model_wrapper,
        categories=categories,
        featureSelector=args.top_n,
        limeSamples=args.lime_sr,
        random_seed=args.seed if not custom_seed else custom_seed,
        success_threshold=args.success_threshold,
        model_batch_size=args.batch_size,
        max_candidates=args.max_candidate,
        logger=None,  # We can add a progress bar here if we want
        modification_rate=args.modify_rate,
        rbo_p=args.rbo_p,
        similarity_measure=args.similarity_measure,
        greedy_search=greedy_search,
    )

    attack_args = AttackArgs(
        num_examples=1,
        random_seed=args.seed if not custom_seed else custom_seed,
        log_to_csv="temp.csv",  # TODO
        checkpoint_interval=250,
        checkpoint_dir="./checkpoints",
        disable_stdout=False,
    )

    attacker = Attacker(attack, Dataset([]), attack_args)

    return attacker


def init_attacker(args: Namespace, model_wrapper: HuggingFaceModelWrapper, categories):
    if args.attack_recipe == "random":
        return generate_attacker(
            args, model_wrapper, categories, custom_seed=None, greedy_search=True
        )

    raise ValueError(f"No attack recipe: {args.attack_recipe}")


def run_experiments(args: Namespace):
    ### Load dataset
    # Load only the test dataset
    # dataset = load_test_data(args.dataset_name, args.seed)
    train_valid_dataset, test_dataset = load_data(args.dataset_name, args.seed)
    dataset = test_dataset
    categories = train_valid_dataset["train"].unique("label")

    # The column name might be different for different dataset and it might
    # be a good idea to select/rename the proper columns or something in
    # load_data.
    # TODO:
    # - Stopwords
    # - maybe move the preprocessing logic into dataset.py
    # - limit number of characters

    def pre_process(sample: dict):
        return AttackedText(sample["text"]), sample["label"]

    data = [pre_process(typing.cast(dict, sample)) for sample in dataset]

    random.seed(args.seed)
    random.shuffle(data)
    data = data[: args.number_of_samples]

    ### Setup model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

    ### Setup attacker
    attacker = init_attacker(args, model_wrapper, categories)
    print(attacker)
