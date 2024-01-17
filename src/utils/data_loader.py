
import scipy 
import numpy as np
def monkeypath_itemfreq(sampler_indices):
   return zip(*np.unique(sampler_indices, return_counts=True))
scipy.stats.itemfreq=monkeypath_itemfreq

from datasets import load_dataset
import textattack

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
    
    from nltk.corpus import stopwords
    return set(stopwords.words('english'))

def process_dataset(dataset, args, stopwords):
    data = []
    for i in range(len(dataset)):
        text = dataset[i].get(args.text_col)
        example = textattack.shared.attacked_text.AttackedText(text)
        num_words_non_stopwords = len([w for w in example._words if w not in stopwords])
        if args.min_length and num_words_non_stopwords < args.min_length:
            continue
        if args.max_length and example.num_words > args.max_length:
            continue
        label = dataset[i].get(args.label_col)
        data.append((example, label))

    categories = list(np.unique([tmp[1] for tmp in data]))
    print("CATEGORIES", categories)

    if args.num > 0:
        rng = np.random.default_rng(seed=args.seed_dataset)
        rng.shuffle(data)
        data = data[:args.num]

    return data, categories