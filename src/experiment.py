# import warnings
# warnings.filterwarnings("ignore")

import torch

import numpy
import scipy 

def monkeypath_itemfreq(sampler_indices):
	return zip(*numpy.unique(sampler_indices, return_counts=True))
scipy.stats.itemfreq=monkeypath_itemfreq

import textattack


# from utils import RANDOM_BASELINE_Attack, ADV_XAI_Attack
from timeit import default_timer as timer

import gc
gc.collect()
torch.cuda.empty_cache()

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:24'
os.environ["TF_GPU_ALLOCATOR"] = 'cuda_malloc_async'

from src.utils.data_loader import *
from src.utils.load_model import *
from src.utils.file_create import *
from src.utils.argparser import *
from src.attack_recipes.gen_attacker import *
from src.attack_recipes.attack import *

if __name__ == "__main__":
    print("Running experiment")
    import tensorflow as tf
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

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
    if args.debug:          
        dataset = dataset.select(range(100))
        print(dataset)
    
    data, categories = process_dataset(dataset, args, stopwords)
	#---------------------------------------------------
    outputName = "output"
    startIndex = 0
    csvName = outputName + str(startIndex) + "_log.csv"
    folderName = "outputName" + str(startIndex)
	#---------------------------------------------------
    
    
    attacker = generate_attacker(args, model_wrapper, categories, csvName)
    
    print(attacker)
    
    results, rbos, sims = perform_attack(data, args, attacker, stopwords, filename)