from src.attack_classes.random_baseline import RandomBaseline
from src.attack_classes.adv_xai_attack import ADV_XAI_Attack

def build_attacker(attack_class, args, model_wrapper, categories, greedy_search=False):
    attack = attack_class.build(
        model_wrapper, 
        categories,
        args.max_candidates,
        args.modify_rate,
        top_n_features = args.top_n,
        greedy_search=greedy_search,
        lime_sr = args.lime_sr, 
        batch_size=args.batch_size 
    )
    attack.cuda_()
    
    return attack

def generate_attacker(args, model_wrapper, categories):    
    if args.method == "random":
        attacker = build_attacker(RandomBaseline, args, model_wrapper, categories, greedy_search=True)
    elif args.method == "xaifooler":
        attacker = build_attacker(ADV_XAI_Attack, args, model_wrapper, categories)
    elif args.method == "inherent":
        # Attack class does not matter 
        attacker1 = build_attacker(RandomBaseline, args, model_wrapper, categories)
        attacker2 = build_attacker(RandomBaseline, args, model_wrapper, categories)

        return attacker1, attacker2
    
    return attacker