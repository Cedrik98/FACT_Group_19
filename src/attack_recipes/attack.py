# import warnings
# warnings.filterwarnings("ignore")

import torch

import numpy
import scipy 

def monkeypath_itemfreq(sampler_indices):
	return zip(*numpy.unique(sampler_indices, return_counts=True))
scipy.stats.itemfreq=monkeypath_itemfreq

# from utils import RANDOM_BASELINE_Attack, ADV_XAI_Attack
from timeit import default_timer as timer

import gc
gc.collect()
torch.cuda.empty_cache()

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:24'
os.environ["TF_GPU_ALLOCATOR"] = 'cuda_malloc_async'

from tqdm import tqdm
import numpy as np
import os

from src.utils.data_loader import *
from src.utils.load_model import *
from src.utils.file_create import *
from src.utils.argparser import *
from src.attack_recipes.gen_attacker import *
from src.evaluation.eval_func import *

def perform_attack(data, args, attacker, stopwords, filename):
    results = []
    rbos = []
    sims = []
    if not args.rerun:
        previous_results = load(filename)
        if previous_results:
            print("LOADED PREVIOUS RESULTS", len(previous_results))
            previous_texts = set([result['example'].text for result in previous_results if not result['log']])
            print(previous_texts)
            results = previous_results
               
    pbar = tqdm(range(0, len(data)), bar_format='{desc:<20}{percentage:3.0f}%|{bar:10}{r_bar}')
    for i in pbar:
        example, label = data[i]
        print("****TEXT*****")
        print("Text", example.text)
        print("Label", label)
        #print("# words (ignore stopwords)", example.num_words_non_stopwords)
        num_words_non_stopwords = len([w for w in example._words if w not in stopwords])
        print("# words (ignore stopwords)", num_words_non_stopwords)

        if not args.rerun and previous_results and example.text in previous_texts:
            print("ALREADY DONE, IGNORE...")
            continue
        # #soft split
        # if args.max_length:
        #     text = " ".join(text.split()[:args.max_length])

        if args.method in set(["xaifooler", "random", "truerandom",'ga']):
            output = attacker.attack.goal_function.get_output(example)
            result = None
            
            #certain malformed instances can return empty dataframes
            
            #result = attacker.attack.attack(example, output)

            try:
                result = attacker.attack.attack(example, output)
            except Exception as e:
                print(f"Error encountered: {e}")
                print("Error generating result")
                results.append({'example': example, 'result': None, 'exp_before': None, 'exp_after': None, 'rbo': None, 'log': 'prediction mismatched'})
                if not args.debug:
                    save(results, filename)
                continue
                
            if result:
                print(result.__str__(color_method="ansi") + "\n")

                sent1 = result.original_result.attacked_text.text
                sent2 = result.perturbed_result.attacked_text.text

                exp1 = attacker.attack.goal_function.generateExplanation(sent1)
                exp2 = attacker.attack.goal_function.generateExplanation(sent2)

            else:
                print("PREDICTION MISMATCHED WITH EXPLANTION")
                results.append({'example': example, 'result': None, 'exp_before': None, 'exp_after': None, 'rbo': None, 'log': 'prediction mismatched'})
                if not args.debug:
                    save(results, filename)
                continue

        elif args.method == "inherent":
            result = None

            sent1 = example.text
            sent2 = example.text

            exp1 = attacker[0].attack.goal_function.generateExplanation(sent1)
            exp2 = attacker[1].attack.goal_function.generateExplanation(sent2)

        print("Base prediction", exp1[1])
        print("Attacked prediction", exp2[1])
        print("sent1", sent1)
        print("sent2", sent2)

        df1 = format_explanation_df(exp1[0], target=exp1[1])
        df2 = format_explanation_df(exp2[0], target=exp2[1])
        print(df1)
        print(df2)

        targetList = df2.get('feature').values
        baseList = df1.get('feature').values

        rboOutput = RBO(targetList, baseList, p=args.rbo_p)
        print("rboOutput", rboOutput)
        rbos.append(rboOutput)

        simOutput = generate_comparative_similarities(result.perturbed_result.attacked_text.text,exp1,exp2)
        print("Comparative Sims", simOutput)
        sims.append(simOutput)
        # pbar.set_description(f"#{i} | Text: {text[:20]}... | RBO Score: {round(rboOutput,2)}")
        pbar.set_description('||Average RBO={}||'.format(np.mean(rbos)))


        pwp = 0
        adjusted_length = 0
        s1 = result.original_result.attacked_text.text.split() 
        s2 = result.perturbed_result.attacked_text.text.split()

        for i in range(len(s1)):
            #print("Comparing: ", s1[i] , s2[i])
            if s1[i][0].isalpha():  
                if s1[i] != s2[i]:
                    pwp += 1
            else:
                #print(s1[i], " is non alphanumeric")
                adjusted_length += 1
        #print(pwp,len(s1),adjusted_length)
        pwp = pwp / (len(s1)-adjusted_length)
        print("Perturbed Word Proportion: ",pwp)

        results.append({'example': example, 'result': result, 'exp_before': exp1, 'exp_after': exp2, 'rbo': rboOutput,'comparativeSims': simOutput, 'log': None,'perturbed_word_proportion': pwp})


        if not args.debug:
            save(results, filename)

    return results, rbos, sims
				