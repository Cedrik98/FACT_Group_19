import argparse

import torch
from textattack.model.wrappers import HuggingFaceDataset, HuggingFaceModelWrapper
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_dataset(dataset_name, _):
    return "a", "b", "c"


def run(args):
    _, dataset_test, categories = load_dataset(args.dataset, args.seed_dataset)
    dataset = HuggingFaceDataset(dataset_test)

    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="XAIFooler reproduction",
        usage="TODO",
    )

    parser.add_argument(
        "--model",
        "-m",
        required=False,
        default="textattack/bert-base-uncased-imdb",
        choices=[
            "textattack/bert-base-uncased-imdb",
            # "distilbert-base-uncased-imdb-saved",
            # "bert-base-uncased-imdb-saved",
            # "roberta-base-imdb-saved",
            # "distilbert-base-uncased-md_gender_bias-saved",
            # "bert-base-uncased-md_gender_bias-saved",
            # "roberta-base-md_gender_bias-saved",
            # "bert-base-uncased-s2d-saved",
            # "distilbert-base-uncased-s2d-saved",
            # "roberta-base-s2d-saved",
        ],
    )
    args = parser.parse_args()

    run(args)
