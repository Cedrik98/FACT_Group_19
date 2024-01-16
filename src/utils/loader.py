
import scipy 
import numpy
def monkeypath_itemfreq(sampler_indices):
   return zip(*numpy.unique(sampler_indices, return_counts=True))
scipy.stats.itemfreq=monkeypath_itemfreq

from datasets import load_dataset
from argparse import ArgumentParser

def load_args():
    parser = ArgumentParser(description='XAIFOOLER')
    parser.add_argument('--lime-sr', type=int, default=None)
    parser.add_argument('--top-n', type=int, default=3)
    parser.add_argument('--model', type=str, default="thaile/distilbert-base-uncased-s2d-saved")
    parser.add_argument('--dataset', type=str, default="s2d")
    parser.add_argument('--label-col', type=str, default="label")
    parser.add_argument('--text-col', type=str, default="text")
    # parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--max-candidate', type=int, default=10)
    parser.add_argument('--success-threshold', type=float, default=0.5)
    parser.add_argument('--rbo-p', type=float, default=0.8)
    parser.add_argument('--num', type=int, default=5)
    parser.add_argument('--modify-rate', type=float, default=0.2)
    parser.add_argument('--max-length', type=int, default=None)
    parser.add_argument('--min-length', type=int, default=10)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--seed-dataset', type=int, default=12)
    parser.add_argument('--method', type=str, default="xaifooler")
    #'xaifooler', ga, random, truerandom
    #parser.add_argument('--search-method',type=str,default = 'default')
    parser.add_argument('--crossover', type=str, default = '1point')
    #uniform, 1point
    parser.add_argument('--parent-selection', type=str, default = 'truncation')
    #roulette, truncation
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--rerun', action='store_true', default=False)
    parser.add_argument('--similarity-measure', type=str, default='rbo')
    #options are rbo, l2, com (general definiton), com_rank_weighted (closest to paper), com_proportional (incomplete)
    #New Similarity Measures are: jaccard, kendall, spearman, append _weighted to each for the weighted version

    args, unknown = parser.parse_known_args()
    return args

def load_dataset_custom(DATASET_NAME, seed):
    print("LOADING", DATASET_NAME)

    if DATASET_NAME == 'imdb':
        dataset = load_dataset(DATASET_NAME)
        dataset_train_valid = dataset['train'].train_test_split(test_size=0.1, stratify_by_column="label", shuffle=True, seed=seed)
        dataset_test = dataset['test']
        categories = dataset_train_valid['train'].unique('label')

    if DATASET_NAME == 'gb':
        DATASET_NAME = "md_gender_bias"
        dataset = load_dataset(DATASET_NAME, 'convai2_inferred', split='train').filter(lambda example: example["binary_label"] in set([0, 1]))
        dataset = dataset.shuffle(seed=seed).select(range(30000))
        dataset = dataset.rename_column("binary_label", "label")
        categories = dataset.unique('label')
        dataset = dataset.train_test_split(test_size=0.2, stratify_by_column="label", shuffle=True, seed=seed)
        dataset_test = dataset['test']
        dataset_train_valid = dataset['train'].train_test_split(test_size=0.1, stratify_by_column="label", shuffle=True, seed=seed)

    elif DATASET_NAME == 's2d':
        DATASET_NAME = 'gretelai/symptom_to_diagnosis'
        dataset = load_dataset("gretelai/symptom_to_diagnosis")
        dataset = dataset.rename_column("output_text", "label")
        dataset = dataset.rename_column("input_text", "text")
        dataset = dataset.class_encode_column("label")
        categories = dataset['train'].unique('label')
        dataset_test = dataset['test']
        dataset_train_valid = dataset['train'].train_test_split(test_size=0.1, stratify_by_column="label", shuffle=True, seed=seed)

    elif DATASET_NAME == 'hate_speech_offensive':
        dataset = load_dataset(DATASET_NAME, split='train').filter(lambda example: example["class"] in set([0, 1]))
        dataset = dataset.rename_column("class", "label")
        categories = dataset.unique('label')
        dataset = dataset.train_test_split(test_size=0.2, stratify_by_column="label", shuffle=True, seed=seed)
        dataset_test = dataset['test']
        dataset_train_valid = dataset['train'].train_test_split(test_size=0.1, stratify_by_column="label", shuffle=True, seed=seed)

    elif DATASET_NAME == 'tweets_hate_speech_detection':
        ### tweets_hate_speech_detection
        dataset = load_dataset(DATASET_NAME, split='train')
        dataset = dataset.rename_column("tweet", "text")
        categories = dataset.unique('label')
        dataset = dataset.train_test_split(test_size=0.2, stratify_by_column="label", shuffle=True, seed=seed)

        dataset_test = dataset['test']
        dataset_train_valid = dataset['train'].train_test_split(test_size=0.1, stratify_by_column="label", shuffle=True, seed=seed)

    return dataset_train_valid, dataset_test, categories

def load_stopwords():
    import nltk
    from nltk.corpus import stopwords
    return set(stopwords.words('english'))