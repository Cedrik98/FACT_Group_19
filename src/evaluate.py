
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy
import scipy 

def monkeypath_itemfreq(sampler_indices):
	return zip(*numpy.unique(sampler_indices, return_counts=True))
scipy.stats.itemfreq=monkeypath_itemfreq

import textattack
import json
from timeit import default_timer as timer

import gc
gc.collect()
torch.cuda.empty_cache()

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:128'
IMDB_DATASET_NAME = "imdb"
MD_GENDER_BIAS_DATASET_NAME = "md_gender_bias"
SYMPTOM_TO_DIAGNOSIS_DATASET_NAME = "symptom_to_diagnosis"
DATASETS = [
    IMDB_DATASET_NAME,
    MD_GENDER_BIAS_DATASET_NAME,
    SYMPTOM_TO_DIAGNOSIS_DATASET_NAME,
]

import numpy as np
import pandas as pd
import nltk
from pathlib import Path
from argparse import ArgumentParser

from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder

from src.utils.file_create import *
from src.evaluation.eval_func import *
from src.dataset.dataset import load_data

def check_condition1(result, df1, df2, configs):
	attacked_text = result.perturbed_result.attacked_text
	modified_index = list(attacked_text.attack_attrs['modified_indices'])
	if modified_index:
		for j in modified_index:
			to_w_j = attacked_text.words[j]
			if to_w_j.lower() not in df1.get('feature') and \
				to_w_j.lower() in df2.get('feature').values[:configs['top_n']]:
				print(f"FAILED! `{to_w_j}`` appears in top_n but was not in the orgiginal text")
				return False
	return True

def check_condition2(exp1, exp2):
	pred_before = exp1[1]
	pred_after = exp2[1]
	return pred_before == pred_after


def topk_intersection(df1, df2, k=None):
	i1 = df1.get('feature').values[:k]
	i2 = df2.get('feature').values[:k]
	return len([x for x in i1 if x in i2])/k
def eval(args, filename=None, use=None):
	if not filename:
		filename = Path(
        f"./results/experiments/{args.model}-{args.dataset}-{args.method}-{args.similarity_measure}/"
    )

	if not use:
		use = UniversalSentenceEncoder(
			threshold=0.840845057,
			metric="angular",
			compare_against_original=False,
			window_size=100,
			skip_text_shorter_than_window=True,
		)

	tmp = {}
	
	_, dataset_test, categories = load_data(args.dataset, args.seed_dataset)
	dataset = textattack.datasets.HuggingFaceDataset(dataset_test)
	dataset = dataset._dataset
	stopwords = set(nltk.corpus.stopwords.words("english"))
	data = []
	for i in range(len(dataset)):
		text = dataset[i].get(args.text_col)
		# TODO stopwords is used here but produces an error
		# example = textattack.shared.attacked_text.AttackedText(text, stopwords=stopwords)
		example = textattack.shared.attacked_text.AttackedText(text)
		# TODO also produces errors
		# if args.min_length and example.num_words_non_stopwords < args.min_length:
		# 	continue
		# if args.max_length and example.num_words > args.max_length:
		# 	continue
		label = dataset[i].get(args.label_col)
		data.append(example)
	
	if args.num > 0:
		rng = np.random.default_rng(seed=args.seed_dataset)
		rng.shuffle(data)
		data = data[:args.num]

	data = set(data)

	with open(f'{filename}/config.json', 'r') as f:
		configs = json.loads(f.read())
	# TODO: How to save results
	results2 = pickle.load(open(f"{filename}/results.pickle", 'rb'))
	
	results = []
	texts = set()
	for a in results2[::-1]:
		if a['example']:
			if a['example'] in data:
				if a['example'].text not in texts:
					results.append(a)
					texts.add(a['example'].text)

	tmp['Total'] = len(results)

	removed = [a for a in results if a['log']]
	print([a['log'] for a in removed])

	results = [a for a in results if not a['log']]
	tmp['Total Adj'] = len(results)

	# preds_before = np.array([a['exp_before'][1] for a in results])
	# preds_after = np.array([a['exp_after'][1] for a in results])
	# idx = np.where(preds_before == preds_after)[0]
	# results = [results[i] for i in idx]

	rbos = []
	sims = []
	l1s = []
	l11s = []
	new_rbos = []
	new_sms = []
	num_replacements = []
	num_errors = 0
	intersections = []

	for item in results:
		result = item['result']
		exp1 = item['exp_before']
		exp2 = item['exp_after']
		df1 = format_explanation_df(exp1[0], target=exp1[1])
		df2 = format_explanation_df(exp2[0], target=exp2[1])
		baseList = df1.get('feature').values
		targetList = df2.get('feature').values

		if not check_condition2(exp1, exp2) and not check_condition1(result, df1, df2, configs):
			num_errors += 1
			continue

		# RBO
		rbo = item['rbo']
		rbos.append(rbo)


		# INTERSECTION
		topkintersect = topk_intersection(df1, df2, k=configs['top_n'])
		intersections.append(topkintersect)

		# SIMILARITY
		if result: # result can be none if running inherent instability
			sent1 = result.original_result.attacked_text.text
			sent2 = result.perturbed_result.attacked_text.text

			if sent1 != sent2:
				emb1, emb2 = use.encode([sent1, sent2])
				sim = use.sim_metric(torch.tensor(emb1.reshape(1,-1)), torch.tensor(emb2.reshape(1,-1)))[0]
			else:
				sim = 1.0
			# print(sim.numpy())
			# print(sent1)
			# print(sent2)
			# print()

		else:
			sim = 1.0
		sims.append(sim)

		if result:
			modified_index = result.perturbed_result.attacked_text.attack_attrs['modified_indices']
			replacement_words = [result.perturbed_result.attacked_text.words[i] for i in modified_index]
			num_replacements.append(len(replacement_words))

		rboOutput = RBO(targetList[:configs['top_n']], baseList[:configs['top_n']], p=1.0)
		new_rbos.append(rboOutput)

		sm = SM(targetList[:configs['top_n']], baseList[:configs['top_n']])
		new_sms.append(sm)

		df2['rank'] = df2.index
		df1['rank'] = df1.index
		
		# print("===============")
		# print(sent1)
		# print("=>", sent2)
		# print(exp1[1])
		# print(exp2[1])
		# print(df1)
		# print(df2)

		rank1 = df1[:configs['top_n']]['rank'].values
		rank2 = df2.set_index('feature').reindex(df1['feature'])[:configs['top_n']]['rank'].values
		l1 = np.sum(np.abs(rank2 - rank1))

		rank1 = df1[:1]['rank'].values
		rank2 = df2.set_index('feature').reindex(df1['feature'])[:1]['rank'].values
		l11= np.sum(np.abs(rank2 - rank1))

		# print(rank1)
		# print(rank2)
		# print(l1)
		# print("===============")

		l1s.append(l1)
		l11s.append(l11)

		# print()
		# print(targetList[:configs['top_n']])
		# print(baseList[:configs['top_n']])
		# print("rboOutput", rboOutput)
		# print("SM", sm)


	# print(sims)
	# print("COMPARE", (np.array(new_sms) == np.array(new_rbos)).mean())
	for threshold in [0.5, 0.6, 0.7]:
		acc = (np.array(rbos) <= threshold).mean()
		tmp['ACC{}'.format(threshold)] = acc


	tmp['Num Errors'] = "{}/{}".format(num_errors, len(results))
	tmp['RBO Avg'] = np.mean(rbos)

	tmp['L1(Top-n) Avg'] = np.mean(l1s)
	tmp['L1(Top-1) Avg'] = np.mean(l11s)

	tmp['SIM Avg'] = np.mean(sims)
	tmp['SIM std'] = np.std(sims)

	tmp['NewRBO Avg'] = np.mean(new_rbos)
	tmp['NewRBO Std'] = np.std(new_rbos)

	tmp['SM Avg'] = np.mean(new_sms)
	tmp['SM std'] = np.mean(new_sms)

	tmp['Rep Avg'] = np.mean(num_replacements)

	tmp["INST(Top-n)"] = np.mean(intersections)


	df = pd.DataFrame.from_dict([tmp])

	return df

if __name__ == "__main__":
    """ Module that performs evaluation on results gathered in experiments. """
    parser = ArgumentParser(description="XAIFOOLER Experiment")
    # All of these should have a help comment!!!
    parser.add_argument("--lime-sr", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--model", type=str, default="thaile/distilbert-base-uncased-imdb-saved")
    parser.add_argument("--dataset", type=str, choices=DATASETS, default="imdb")
    parser.add_argument("--label-col", type=str, default="label")
    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)  # TODO: change back to 512
    parser.add_argument("--max-candidate", type=int, default=10)
    parser.add_argument("--success-threshold", type=float, default=0.5)
    parser.add_argument("--rbo-p", type=float, default=0.8)
    parser.add_argument("--num", type=int, default=5)
    parser.add_argument("--modify-rate", type=float, default=0.2)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--min-length", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--seed-dataset", type=int, default=12)
    parser.add_argument(
        "--method", type=str, choices=["ga", "random", "truerandom"], default="random"
    )
    parser.add_argument("--search-method", type=str, default="default")
    parser.add_argument(
        "--crossover", type=str, choices=["uniform", "1point"], default="1point"
    )
    parser.add_argument(
        "--parent-selection",
        type=str,
        choices=["roulette", "truncation"],
        default="truncation",
    )
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--rerun", action="store_true", default=False)
    parser.add_argument(
        "--similarity-measure",
        type=str,
        choices=[
            "rbo",
            "l2",
            "com",
            "com_rank_weighted",
            "com_proportional",
            "jaccard",
            "kendall",
            "spearman",
            "jaccard_weighted",
            "kendall_weighted",
            "spearman_weighted",
        ],
        default="rbo",
    )
    parser.add_argument(
        "--number-of-samples",
        type=int,
        default=25000,
        help="Number of datapoints sampled from the dataset",
        required=False,
    )
    print("works")
    print(eval(parser))