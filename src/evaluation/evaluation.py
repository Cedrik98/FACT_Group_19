import numpy as np
import torch

from scipy import stats


def compute_abs(exp_df1, exp_df2, top_n):
    rank1 = exp_df1[:top_n]["rank"].values
    rank2 = (
        exp_df2.set_index("feature").reindex(exp_df1["feature"])[:top_n]["rank"].values
    )
    l1 = np.sum(np.abs(rank2 - rank1))
    return l1


def compute_ins(df1, df2, top_n: int = 1):
    i1 = df1.get("feature").values[:top_n]
    i2 = df2.get("feature").values[:top_n]
    return len([x for x in i1 if x in i2]) / top_n


def compute_rc(list1, list2, top_n):
    coef, p = stats.spearmanr(list1[:top_n], list2[:top_n])
    return 1 - max(0, coef)


def compute_sim(result, encoder):
    if not result:
        return 1.0

    sent1 = result.original_result.attacked_text.text
    sent2 = result.perturbed_result.attacked_text.text

    if sent1 == sent2:
        return 1.0

    emb1, emb2 = encoder.encode([sent1, sent2])
    return encoder.sim_metric(
        torch.tensor(emb1.reshape(1, -1)), torch.tensor(emb2.reshape(1, -1))
    )[0]
