"""
This project tries to reproduce results from the paper
'Are Your Explanations Reliable?' Investigating the Stability of LIME in
Explaining Text Classifiers by Marrying XAI and Adversarial Attack.
"""
import argparse
from argparse import Namespace


def run(args: Namespace):
    """Entry point to the program."""
    # These import are here because they are very slow. This means that they
    # would take a long time loading even when just using the -h flag. To
    # prevent this they are not loaded in at the top-level.
    import torch
    from textattack.models.wrappers import HuggingFaceModelWrapper
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.dataset import load_test_data

    # Load only the test dataset
    dataset = load_test_data(args.dataset_name, args.seed)

    # Setup model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

    # TODO:
    # - Functionality from run() might need to be extracted.
    # - There needs to be a differentiation between running and training.
    print(dataset, model_wrapper)


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

    run(parser.parse_args())
