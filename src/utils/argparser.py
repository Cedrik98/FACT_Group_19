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
    # parser.add_argument('--search-method',type=str,default = '"default"')
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