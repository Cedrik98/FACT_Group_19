import typing
from timeit import default_timer as timer
import nltk
import textattack
from datasets.arrow_dataset import Dataset
from argparse import ArgumentParser, Namespace
from textattack.models.wrappers.huggingface_model_wrapper import (
    HuggingFaceModelWrapper,
)



from src.dataset import DATASETS, load_data, process_experiment_data
from src.utils.model import MODELS, load_trained_model_and_tokenizer
from src.utils.constants import HF_ACCOUNT
from src.attack.gen_attacker import generate_attacker
from src.attack.perform_attack import perform_attack

def run_experiment(args: Namespace):

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

    stopwords = set(nltk.corpus.stopwords.words("english"))

    # Preprocess data
    data, categories = process_experiment_data(dataset, args, stopwords)    
    
    # Generate attacker
    attack = generate_attacker(args, model_wrapper, categories)

    # Perform attack
    start = timer()
    result = perform_attack(data, attack, args)
    end = timer()

    print("Experiment took...", end - start)
    print(result)
    

if __name__ == "__main__":    
    parser = ArgumentParser(description="XAIFOOLER Experiment")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=DATASETS,
        default="md_gender_bias",
        help="Dataset to use for experiment",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed to use",
    )
    parser.add_argument(
        "--number-of-samples",
        type=int,
        default=25000,
        help="Number of datapoints sampled from the dataset",
        required=False,
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
        "--num",
        type=int,
        default=5,
        help="The number of documents to attack (Reccomended due to time only"
        "to choose a small subset of the included datasets)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=MODELS,
        default="distilbert-base-uncased",
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
        "--max-candidates",
        type=int,
        default=10,
        help="Max Candidates - The number of nearest neighbors calculated"
        "for word replacement",
    )
    parser.add_argument(
        "--modify-rate",
        type=float,
        default=0.2,
        help="Modify Rate - The maximum percentage of possible perturbed words"
        "as a percentage of the document",
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
        "--top-n",
        type=int,
        default=3,
        help="The top-n feature threshold - Which features are required"
        "to be held constant between each iteration of the search",
    )
    parser.add_argument(
        "--lime-sr",
        type=int,
        default=10,
        help="Lime sampling rate",
    )

    parser.add_argument("--debug", action="store_true", default=False)

    run_experiment(parser.parse_args())