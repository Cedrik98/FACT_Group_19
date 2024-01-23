from typing import Tuple

from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODELS = [
    "distilbert-base-uncased",
    "bert-base-uncased",
    "roberta-base",
    "gpt2",
]


def load_trained_model_and_tokenizer(
    model_name, dataset_name, hf_account
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    print(hf_account, model_name, dataset_name)
    repo_name = f"{hf_account}/{model_name}-{dataset_name}-trained"
    print(repo_name)
    model = AutoModelForSequenceClassification.from_pretrained(repo_name)
    tokenizer = AutoTokenizer.from_pretrained(repo_name)
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
