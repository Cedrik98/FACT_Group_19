from argparse import Namespace

import numpy as np
import textattack

# from datasets import Dataset
from datasets.arrow_dataset import Dataset


def process_experiment_data(dataset: Dataset, args: Namespace, stopwords):
    data = []
    for i in range(len(dataset)):
        text = dataset[i].get(args.text_col)
        if args.debug:
            example = textattack.shared.attacked_text.AttackedText(text)

        else:
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
    print(args.num)
    if args.num > 0:
        rng = np.random.default_rng(seed=args.seed)
        rng.shuffle(data)
        data = data[: args.num]
    print("HELLO")
    print(len(data))
    return data, categories
