import eli5
import numpy as np

def format_explanation_df(explanation, target=-1):
    df = eli5.format_as_dataframes(explanation)["targets"]
    df["abs_weight"] = np.abs(df["weight"])
    df = df.sort_values(by=["abs_weight"], ascending=False)
    if target > -1:
        idx = df.apply(lambda x: x["target"] == target, axis=1)
        df = df[idx]
        df = df.reset_index(drop=True)
    return df

