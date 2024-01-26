# from timeit import default_timer as timer
import nltk
from textattack.constraints.semantics.sentence_encoders import (
    UniversalSentenceEncoder,
)

from src.evaluation.evaluation import (
    compute_abs,
    compute_ins,
    compute_rc,
    compute_sim,
)
from src.utils.file_create import format_explanation_df, load


def dirty_fix():
    """Has to be done because of dependency problems."""
    import numpy
    import scipy

    def monkeypath_itemfreq(sampler_indices):
        return zip(*numpy.unique(sampler_indices, return_counts=True))

    scipy.stats.itemfreq = monkeypath_itemfreq


dirty_fix()


def load_previous_results(filename):
    """Loads in previous results."""
    previous_results = load(filename)

    if previous_results:
        print("LOADED PREVIOUS RESULTS", len(previous_results))

        previous_texts = set(
            [
                result["example"].text
                for result in previous_results
                if not result["log"]
            ]
        )

        return previous_results, previous_texts


def do_single_attack(example, attacker, args):
    """Performs a single attack."""
    if args.method == "inherent":
        result = None

        sent1 = example.text
        sent2 = example.text

        exp1 = attacker[0].attack.goal_function.generateExplanation(
            sent1,
            custom_n_samples=args.lime_sr,
        )
        exp2 = attacker[1].attack.goal_function.generateExplanation(
            sent2,
            custom_n_samples=args.lime_sr,
        )

        return sent1, exp1, sent2, exp2, None

    elif args.method in set(["xaifooler", "random", "truerandom", "ga"]):
        output = attacker.attack.goal_function.get_output(example)
        result = None

        try:  # certain malformed instances can return empty dataframes
            result = attacker.attack.attack(example, output)
        except Exception as e:
            print("Exception!!!!!!!!!!!!")
            print(e)
            return None

        if result:
            sent1 = result.original_result.attacked_text.text
            sent2 = result.perturbed_result.attacked_text.text

            exp1 = attacker.attack.goal_function.generateExplanation(
                sent1,
                custom_n_samples=args.lime_sr,
            )
            exp2 = attacker.attack.goal_function.generateExplanation(
                sent2,
                custom_n_samples=args.lime_sr,
            )

            return sent1, exp1, sent2, exp2, result
        else:
            print("NO RESULT!!!!!!!!!!!")
            return None


def init_universal_sentence_encoder():
    return UniversalSentenceEncoder(
        threshold=0.840845057,
        metric="angular",
        compare_against_original=False,
        window_size=100,
        skip_text_shorter_than_window=True,
    )


def perform_attack(data, args, attacker, stopwords, filename):
    results = []
    previous_results = None
    previous_texts = None
    universal_sentence_encoder = init_universal_sentence_encoder()
    stopwords = set(nltk.corpus.stopwords.words("english"))

    if not args.rerun:
        previous_results, previous_texts = load_previous_results(filename)
        results = previous_results

    for i, sample in enumerate(data):
        example, label = sample
        num_words_non_stopwords = len(
            [w for w in example._words if w not in stopwords]
        )

        print("--------------------")
        print(f"Sample numnber: {i}")
        print(f"Text: {example.text[:30]}...")
        print(f"Label: {label}")
        print(f"#words: {num_words_non_stopwords} (ignore stopwords)")

        if (
            not args.rerun
            and previous_results
            and example.text in previous_texts
        ):
            print("ALREADY DONE, IGNORE...")

        res = do_single_attack(example, attacker, args)

        if not res:
            results.append(
                {
                    "example_before": example,
                    "example_after": None,
                    "exp_before": None,
                    "exp_after": None,
                    "abs_score": None,
                    "ins_score": None,
                    "rc_score": None,
                    "sim_score": None,
                }
            )
            continue

        sent1, exp1, sent2, exp2, result = res

        df1 = format_explanation_df(exp1[0], target=exp1[1])
        df2 = format_explanation_df(exp2[0], target=exp2[1])
        base_list = df1.get("feature").values
        target_list = df2.get("feature").values

        print("DATAFRAMES")
        print("orig")
        print(df1)
        print("perturbed")
        print(df2)

        # TODO: what is top_n?
        abs_score = compute_abs(df1, df2, top_n=args.top_n)
        ins_score = compute_ins(df1, df2, top_n=args.top_n)
        rc_score = compute_rc(target_list, base_list, top_n=args.top_n)
        sim_score = compute_sim(result, encoder=universal_sentence_encoder)

        results.append(
            {
                "example_before": sent1,
                "example_after": sent2,
                "exp_before": exp1,
                "exp_after": exp2,
                "abs_score": abs_score,
                "ins_score": ins_score,
                "rc_score": rc_score,
                "sim_score": sim_score,
            }
        )

    return results
