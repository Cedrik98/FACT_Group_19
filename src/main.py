# import warnings
# warnings.filterwarnings("ignore")

import torch
import math
import numpy
import scipy 

def monkeypath_itemfreq(sampler_indices):
	return zip(*numpy.unique(sampler_indices, return_counts=True))
scipy.stats.itemfreq=monkeypath_itemfreq

import textattack
import transformers

# from utils import RANDOM_BASELINE_Attack, ADV_XAI_Attack
from timeit import default_timer as timer

import gc
gc.collect()
torch.cuda.empty_cache()

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:24'
os.environ["TF_GPU_ALLOCATOR"] = 'cuda_malloc_async'

import pickle
from tqdm import tqdm
import numpy as np
import os
import json
import time

from argparse import ArgumentParser

from utils.data_loader import *
from utils.load_model import *
from utils.file_create import *
from utils.argparser import *
from attack_recipes.gen_attacker import *
from attack_recipes.attack import *

if __name__ == "__main__":
    args = load_args()
    filename = generate_filename(args)

    setup_output_environment(filename, args)

    models = ['distilbert-base-uncased-imdb-saved',
        'bert-base-uncased-imdb-saved',
        'roberta-base-imdb-saved',
        'distilbert-base-uncased-md_gender_bias-saved',
        'bert-base-uncased-md_gender_bias-saved',
        'roberta-base-md_gender_bias-saved',
        'bert-base-uncased-s2d-saved',
        'distilbert-base-uncased-s2d-saved',
        'roberta-base-s2d-saved']

    model, tokenizer, model_wrapper = load_model_and_tokenizer(args.model, args.max_length, args.device, models)
    stopwords = load_stopwords()

    _, dataset_test, categories = load_dataset_custom(args.dataset, args.seed_dataset)
    dataset = textattack.datasets.HuggingFaceDataset(dataset_test)
   
    dataset = dataset._dataset
    data, categories = process_dataset(dataset, args, stopwords)
	#---------------------------------------------------
    outputName = "output"
    startIndex = 0
    csvName = outputName + str(startIndex) + "_log.csv"
    folderName = "outputName" + str(startIndex)
	#---------------------------------------------------
    

    attacker = generate_attacker(args, model_wrapper, categories, csvName)
    results, rbos, sims = perform_attack(data, args, attacker, stopwords, filename)