"""
This project tries to reproduce results from the paper
'Are Your Explanations Reliable?' Investigating the Stability of LIME in
Explaining Text Classifiers by Marrying XAI and Adversarial Attack.
"""
import argparse
from argparse import Namespace

# from xai_fooler.src.dataset import load_test_data
# from textattack.models.wrappers import HuggingFaceModelWrapper
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# import torch


def run(args: Namespace):
    """Entry point to the program."""
    from src.dataset import load_test_data

    dataset = load_test_data(args.dataset_name, args.seed)
    print(dataset)
    print(args)


# model = AutoModelForSequenceClassification.from_pretrained(args.model)
# tokenizer = AutoTokenizer.from_pretrained(args.model)
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)
#
# model_wrapper = HuggingFaceModelWrapper(model, tokenizer)


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
