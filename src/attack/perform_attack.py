import numpy as np
from tqdm import tqdm
from textattack.attack_results import SuccessfulAttackResult
from src.evaluation.evaluation import (
    compute_abs,
    compute_ins,
    compute_rc,
    compute_sim,
)
from textattack.constraints.semantics.sentence_encoders import (
    UniversalSentenceEncoder,
)

from src.utils.format import format_explanation_df

def init_universal_sentence_encoder():
    return UniversalSentenceEncoder(
        threshold=0.840845057,
        metric="angular",
        compare_against_original=False,
        window_size=15,
        skip_text_shorter_than_window=True,    )


def perform_attack(data, attack, args):    
    result = []
    universal_sentence_encoder = init_universal_sentence_encoder()
    count_suc = 0
    count_fail = 0
    count_error = 0
    pbar = tqdm(range(0, len(data)), bar_format='{desc:<20}{percentage:3.0f}%|{bar:10}{r_bar}')
    for i, sample in enumerate(data):
        pbar.update(1)
        example, label = sample        
        results = None
        if args.method == "random" or args.method == "xaifooler":
            init_label_pred = attack.goal_function.get_output(example)
            try:
                results = attack.attack(example, init_label_pred)
            except Exception as e:
                # print("==================================")
                # print("ATTACK FAILED")
                # print("Probably LIME returned no explanation")
                # print(e)
                # print("==================================")
                count_error += 1
                continue
            if isinstance(results, SuccessfulAttackResult):
                count_suc += 1
                # print("==================================")
                # print("ATTACK SUCCESFULL")
                # print("==================================")
                
                sent1 = results.original_text()
                sent2 = results.perturbed_text()            
                
                # Authors called Lime an additional time
                exp1 = attack.goal_function.base_explanation[1:]            
                exp2 = attack.goal_function.final_explanation
                
                df1 = format_explanation_df(exp1[0], target=exp1[1])
                df2 = format_explanation_df(exp2[0], target=exp2[1])
                base_list = df1.get("feature").values
                target_list = df2.get("feature").values

                # print("DATAFRAMES")
                # print("orig")
                # print(df1)
                # print("perturbed")
                # print(df2)

                abs_score = compute_abs(df1, df2, top_n=args.top_n)
                ins_score = compute_ins(df1, df2, top_n=args.top_n)
                rc_score = compute_rc(target_list, base_list, top_n=args.top_n)
                sim_score = compute_sim(sent1, sent2, encoder=universal_sentence_encoder)

                result.append(
                    {
                        # "example_before": sent1,
                        # "example_after": sent2,
                        # "exp_before": exp1,
                        # "exp_after": exp2,
                        "abs_score": abs_score,
                        "ins_score": ins_score,
                        "rc_score": rc_score,
                        "sim_score": sim_score,
                    }
                )
            else:
                # print(results)
                count_fail += 1
                # print("==================================")
                # print("ATTACK FAILED")
                # print("==================================")

        elif args.method == "inherent":
            sent1 = example.text
            sent2 = example.text
            try:
                exp1 = attack[0].goal_function.generate_explanation(
                    sent1,
                    random_seed = np.random.choice(1000),
                    custom_n_samples=args.lime_sr,
                )                
                exp2 = attack[1].goal_function.generate_explanation(
                    sent2,
                    random_seed = np.random.choice(1000),
                    custom_n_samples=args.lime_sr,
                )
            except Exception as e:
                count_error += 1
                # print("==================================")
                # print("ATTACK FAILED")
                # print("Something with Lime no solution yet")
                # print(e)
                # print("==================================")
                count_error += 1
                continue
            
            df1 = format_explanation_df(exp1[0], target=exp1[1])
            df2 = format_explanation_df(exp2[0], target=exp2[1])
            
            base_list = df1.get("feature").values
            target_list = df2.get("feature").values
            if base_list != [] and target_list != []: 
                count_suc += 1           
                # print(base_list)
                # print(target_list)
                # print("DATAFRAMES")
                # print("orig")
                # print(df1)
                # print("perturbed")
                # print(df2)

                abs_score = compute_abs(df1, df2, top_n=args.top_n)
                ins_score = compute_ins(df1, df2, top_n=args.top_n)
                rc_score = compute_rc(target_list, base_list, top_n=args.top_n)
                sim_score = compute_sim(sent1, sent2, encoder=universal_sentence_encoder)

                result.append(
                    {
                        # "example_before": sent1,
                        # "example_after": sent2,
                        # "exp_before": exp1,
                        # "exp_after": exp2,
                        "abs_score": abs_score,
                        "ins_score": ins_score,
                        "rc_score": rc_score,
                        "sim_score": sim_score,
                    }
                )
            else:
                count_fail += 1

    print(f"\nATTACK COMPLETE: \nError Count: {count_error}\nFailure Count: {count_fail}\nSuccess Count: {count_suc}")

    if count_suc == 0:
        return "FAILED EXPERIMENT"
    else:
        average_results = {
            "avg_abs_score": sum(r["abs_score"] for r in result) / len(result),            
            "avg_rc_score": sum(r["rc_score"] for r in result) / len(result),
            "avg_ins_score": sum(r["ins_score"] for r in result) / len(result),
            "avg_sim_score": sum(r["sim_score"] for r in result) / len(result)
        }

    return average_results
            
        