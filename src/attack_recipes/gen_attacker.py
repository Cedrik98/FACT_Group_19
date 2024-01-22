import numpy as np
import textattack

from src.attack_recipes.adversarial_xai_rbo import ADV_XAI_Attack
from src.attack_recipes.random_baseline import RANDOM_BASELINE


def build_attacker(
    ATTACK_CLASS,
    args,
    model_wrapper,
    categories,
    csvName,
    custom_seed=None,
    greedy_search=True,
):
    if args.lime_sr is not None:
        samples = args.lime_sr
    elif args.dataset == "imdb":
        samples = 4500
    elif args.dataset == "gb":
        samples = 1500
    elif args.dataset == "s2d":
        samples = 2500
    else:
        samples = 5000
    pbar = None
    attack = ATTACK_CLASS.build(
        model_wrapper,
        categories=categories,
        featureSelector=args.top_n,
        limeSamples=samples,
        random_seed=args.seed if not custom_seed else custom_seed,
        success_threshold=args.success_threshold,
        model_batch_size=args.batch_size,
        max_candidates=args.max_candidate,
        logger=pbar if args.debug else None,
        modification_rate=args.modify_rate,
        rbo_p=args.rbo_p,
        similarity_measure=args.similarity_measure,
        greedy_search=greedy_search,
        search_method=args.method,
        crossover=args.crossover,
        parent_selection=args.parent_selection,
    )

    attack_args = textattack.AttackArgs(
        num_examples=1,
        random_seed=args.seed if not custom_seed else custom_seed,
        log_to_csv=csvName,
        checkpoint_interval=250,
        checkpoint_dir="./checkpoints",
        disable_stdout=False,
    )

    attacker = textattack.Attacker(attack, textattack.datasets.Dataset([]), attack_args)

    return attacker


def generate_attacker(
    args,
    model_wrapper,
    categories,
    csvName,
):
    if args.method == "xaifooler":
        attacker = build_attacker(
            ADV_XAI_Attack, args, model_wrapper, categories, csvName, custom_seed=None
        )

    elif args.method == "inherent":
        attacker1 = build_attacker(
            ADV_XAI_Attack,
            args,
            model_wrapper,
            categories,
            csvName,
            custom_seed=np.random.choice(1000),
        )
        attacker2 = build_attacker(
            ADV_XAI_Attack,
            args,
            model_wrapper,
            categories,
            csvName,
            custom_seed=np.random.choice(1000),
        )

        return attacker1, attacker2
    elif args.method == "random":
        attacker = build_attacker(
            RANDOM_BASELINE,
            args,
            model_wrapper,
            categories,
            csvName,
            custom_seed=None,
            greedy_search=True,
        )

    elif args.method == "truerandom":
        attacker = build_attacker(
            RANDOM_BASELINE,
            args,
            model_wrapper,
            categories,
            csvName,
            custom_seed=None,
            greedy_search=False,
        )

    elif args.method == "ga":
        attacker = build_attacker(
            ADV_XAI_Attack, args, model_wrapper, categories, csvName, custom_seed=None
        )

    return attacker
