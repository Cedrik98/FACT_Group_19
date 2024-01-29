""" Module that fine tunes language models on specific tasks. """
from argparse import ArgumentParser, Namespace
from typing import Any

from src.utils.constants import HF_ACCOUNT, TRAIN_LOGGING_PATH
from src.dataset import DATASETS
from src.utils.model import MODELS, save_model_and_tokenizer


def compute_metrics():
    """Returns evaluation function."""
    import numpy as np

    import evaluate

    metric_acc = evaluate.loading.load("accuracy")
    metric_f1 = evaluate.loading.load("f1")

    def evaluation_func(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        return {
            "accuracy": metric_acc.compute(predictions=predictions, references=labels),
            "f1": metric_f1.compute(
                predictions=predictions, references=labels, average="weighted"
            ),
        }

    return evaluation_func


def train(args: Namespace):
    """Trains a model/dataset combination."""
    # TODO: look at this
    # from transformers import EarlyStoppingCallback
    import gc
    import os
    from pathlib import Path

    import numpy as np
    import torch
    import transformers
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    from src.dataset import load_data

    print(f"\nTRAINING: {args.model} {args.dataset}")
    print(f"ARGS: {args}\n")

    # Collect garbage using garbage collection.
    gc.collect()
    torch.cuda.empty_cache()

    # Prevent memory fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    model_checkpoints_dir = Path(f"{args.model}-{args.dataset}-checkpoints")
    model_trained_dir = Path(f"{args.model}-{args.dataset}-trained")
    logs_dir = Path(f"{args.model}-{args.dataset}-trained")

    # Set paths for storing information
    train_logging_path = Path(TRAIN_LOGGING_PATH)
    model_checkpoints_path = train_logging_path / model_checkpoints_dir
    model_trained_path = train_logging_path / model_trained_dir
    logs_path = train_logging_path / logs_dir

    # Change verbosity level to prevent warnings
    if args.debug:
        transformers.logging.set_verbosity_warning()
    else:
        transformers.logging.set_verbosity_error()

    # Load dataset
    train_valid_dataset, test_dataset, categories = load_data(
        args.dataset, seed=args.seed, number_of_samples=args.number_of_samples
    )
    train_dataset = train_valid_dataset["train"]
    valid_dataset = train_valid_dataset["test"]

    # Select only a few datapoints to speedup training during debugging
    if args.debug:
        indices = np.arange(16)
        train_dataset = train_dataset.select(indices)
        valid_dataset = valid_dataset.select(indices)
        test_dataset = test_dataset.select(indices)

    # Initalize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    if args.model == "gpt2":
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(categories)
    )

    if args.model == "gpt2":
        model.config.pad_token_id = model.config.eos_token_id

    # Tokenize the data
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            add_special_tokens=True,
            # Docs: pad to a length specified by the max_length argument or
            # the maximum length accepted by the model if no max_length is provided
            # (max_length=None). Padding will still be applied if you only provide a
            # single sequence.
            max_length=args.max_length if args.max_length else None,
            padding="max_length",
            truncation=True,
        )

    tokenized_train_dataset: Any = train_dataset.map(
        tokenize_function, batched=True, desc="Tokenizing train data"
    )
    tokenized_valid_dataset: Any = valid_dataset.map(
        tokenize_function, batched=True, desc="Tokenizing validation data"
    )
    tokenized_test_dataset: Any = test_dataset.map(
        tokenize_function, batched=True, desc="Tokenizing test data"
    )

    print(
        f"Number of train samples:\t{len(tokenized_train_dataset)}\n"
        f"Number of validation samples:\t{len(tokenized_valid_dataset)}\n"
        f"Number of test samples:\t\t{len(tokenized_test_dataset)}"
    )

    # Lower these to speedup training
    if args.debug:
        args.epochs = 3
        args.batch_size = 8

    # Train models
    training_args = TrainingArguments(
        output_dir=str(model_checkpoints_path),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=100,
        weight_decay=args.weight_decay,
        logging_dir=str(logs_path),
        logging_steps=10,
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        compute_metrics=compute_metrics(),
        # callbacks = [EarlyStoppingCallback(early_stopping_patience=1)]
    )
    trainer.train()

    # Evaluate models
    result = trainer.evaluate(tokenized_test_dataset)
    print(result)

    save_model_and_tokenizer(
        model, tokenizer, model_trained_path, model_trained_dir, hf_account=HF_ACCOUNT
    )


def run(args: Namespace):
    """Entry point for evaluation."""
    import itertools

    if args.all:
        for model, dataset in list(itertools.product(MODELS, DATASETS)):
            args.model = model
            args.dataset = dataset

            if model == "bert-base-uncased":
                args.batch_size = 128

            train(args)
    else:
        train(args)


if __name__ == "__main__":
    parser = ArgumentParser(description="XAIFOOLER Training")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        choices=MODELS,
        default="distilbert-base-uncased",
        help="Model to train",
        required=False,
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        choices=DATASETS,
        default="imdb",
        help="Dataset to train on",
        required=False,
    )
    parser.add_argument(
        "--batch-size",
        "-bs",
        type=int,
        default=512,
        help="Batch size to use during training",
        required=False,
    )
    parser.add_argument(
        "--max-length",
        "-ml",
        type=int,
        default=250,
        help="Length to which to truncate if it is set, can be set to None for"
        " maximum length excepted by the model",
        required=False,
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=3,
        help="Number of epochs to train for",
        required=False,
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed to use during training",
        required=False,
    )
    parser.add_argument(
        "--weight-decay",
        "-wd",
        type=float,
        default=0.01,
        help="Weight decay during training",
        required=False,
    )
    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="Indicates if debuging or not",
        required=False,
    )
    parser.add_argument(
        "--number-of-samples",
        type=int,
        default=25000,
        help="Number of datapoints sampled from the dataset",
        required=False,
    )
    parser.add_argument(
        "--all",
        type=bool,
        default=False,
        help="Trains all combinations of models/datasets",
        required=False,
    )
    run(parser.parse_args())
