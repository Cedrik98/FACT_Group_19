import scipy 
import numpy
def monkeypath_itemfreq(sampler_indices):
   return zip(*numpy.unique(sampler_indices, return_counts=True))
scipy.stats.itemfreq=monkeypath_itemfreq
import eli5
import numpy as np
import pickle

def save(results, filename):
	with open('{}/results.pickle'.format(filename), 'wb') as handle:
		pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
		print("UPDATED TO", filename)

def load(filename):
	results = None
	try:
		results = pickle.load(open(f"{filename}/results.pickle", 'rb'))
	except:
		pass
	return results

def generate_filename(args):
    if args.method == "xaifooler":
        if args.similarity_measure == "rbo":
            filename = f"./results/dataset-{args.dataset.replace('/','-')}_model-{args.model.replace('/','-')}_s-{args.batch_size}_k-{args.max_candidate}_n-{args.top_n}_-sr-{args.lime_sr}_threshold-{args.success_threshold}_seed-{args.seed}{args.seed_dataset}_modifyrate-{args.modify_rate}_RBOrate-{args.rbo_p}/"
        else:
            filename = f"./results/{args.similarity_measure.upper()}_dataset-{args.dataset.replace('/','-')}_model-{args.model.replace('/','-')}_s-{args.batch_size}_k-{args.max_candidate}_n-{args.top_n}_-sr-{args.lime_sr}_threshold-{args.success_threshold}_seed-{args.seed}{args.seed_dataset}_modifyrate-{args.modify_rate}_RBOrate-{args.rbo_p}/"
    elif args.method == "ga":
        if args.similarity_measure == "rbo":
            filename = f"./results/GA/{args.crossover}--{args.parent_selection}--dataset-{args.dataset.replace('/','-')}_model-{args.model.replace('/','-')}_s-{args.batch_size}_k-{args.max_candidate}_n-{args.top_n}_-sr-{args.lime_sr}_threshold-{args.success_threshold}_seed-{args.seed}{args.seed_dataset}_modifyrate-{args.modify_rate}_RBOrate-{args.rbo_p}/"
        else:
            filename = f"./results/GA/{args.crossover}--{args.parent_selection}--{args.similarity_measure.upper()}_dataset-{args.dataset.replace('/','-')}_model-{args.model.replace('/','-')}_s-{args.batch_size}_k-{args.max_candidate}_n-{args.top_n}_-sr-{args.lime_sr}_threshold-{args.success_threshold}_seed-{args.seed}{args.seed_dataset}_modifyrate-{args.modify_rate}_RBOrate-{args.rbo_p}/"

    else:
        filename = f"./results/{args.method.upper()}BASELINE_dataset-{args.dataset.replace('/','-')}_model-{args.model.replace('/','-')}_s-{args.batch_size}_k-{args.max_candidate}_n-{args.top_n}_-sr-{args.lime_sr}_threshold-{args.success_threshold}_seed-{args.seed}{args.seed_dataset}_modifyrate-{args.modify_rate}_RBOrate-{args.rbo_p}/"
    return filename

def format_explanation_df(explanation, target=-1):	
    df = eli5.format_as_dataframes(explanation)['targets']	
    df['abs_weight'] = np.abs(df['weight'])	
    df = df.sort_values(by=['abs_weight'], ascending=False)	
    if target > -1:	
        idx = df.apply(lambda x: x['target'] == target, axis=1)	
        df = df[idx]	
        df = df.reset_index(drop=True)	
    return df