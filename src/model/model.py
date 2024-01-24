from typing import Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODELS = [
    "distilbert-base-uncased",
    "bert-base-uncased",
    "roberta-base",
    "gpt2",
]


def load_trained_model_and_tokenizer(
    model_name, dataset_name, hf_account, categories
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    repo_name = f"{hf_account}/{model_name}-{dataset_name}-trained"
    model = AutoModelForSequenceClassification.from_pretrained(repo_name, num_labels=len(categories))
    tokenizer = AutoTokenizer.from_pretrained(repo_name)

    # Put model on device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, tokenizer


def save_model_and_tokenizer(
    model, tokenizer, model_trained_path, model_trained_dir, hf_account
):
    """Saves model and tokenizer locally and uploads them to huggingface."""
    # Store trained models
    tokenizer.save_pretrained(model_trained_path)
    model.save_pretrained(model_trained_path)

    # Push everything to huggingface
    tokenizer.push_to_hub(f"{hf_account}/{model_trained_dir}")
    model.push_to_hub(f"{hf_account}/{model_trained_dir}")
