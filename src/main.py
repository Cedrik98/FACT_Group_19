"""
This project tries to reproduce results from the paper
'Are Your Explanations Reliable?' Investigating the Stability of LIME in
Explaining Text Classifiers by Marrying XAI and Adversarial Attack.
"""
import argparse
from argparse import Namespace

# TODO:
# - Train model vs xaifooler stuff
# - Clean everything in attack_recipes folder


def run(args: Namespace):
    """Entry point to the program."""
    # This import is here because it is very slow. This means that it
    # would take a long time loading even when just using the -h flag. To
    # prevent this it is not loaded in at the top-level.

    from src.temp import run_experiments

    run_experiments(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="XAIFooler reproduction",
        usage="TODO",
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        choices=[
            "textattack/bert-base-uncased-imdb",
            # "roberta-base-md_gender_bias-saved",
            # "bert-base-uncased-s2d-saved",
            # "distilbert-base-uncased-s2d-saved",
            # "roberta-base-s2d-saved",
        ],
        default="textattack/bert-base-uncased-imdb",
        help="The model experiments should be applied to",
    )
    parser.add_argument(
        "--dataset-name",
        "-dn",
        type=str,
        choices=["imdb"],
        default="imdb",
        help="The name of the dataset to use",
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42, help="The seed to use for reproducability"
    )
    parser.add_argument(
        "--number-of-samples",
        "-nof",
        type=int,
        default=100,
        help="The number of datapoints that should be used",
    )
    parser.add_argument(
        "--attack-recipe",
        "-pm",
        type=str,
        choices=["random"],
        default="random",
        help="The method for generating new samples from the base sample",
    )

    run(parser.parse_args())
