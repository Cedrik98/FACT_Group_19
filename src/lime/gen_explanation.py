import textattack
def dirty_fix():
    """Has to be done because of dependency problems."""
    import numpy
    import scipy

    def monkeypath_itemfreq(sampler_indices):
        return zip(*numpy.unique(sampler_indices, return_counts=True))

    scipy.stats.itemfreq = monkeypath_itemfreq


dirty_fix()
from eli5.lime import TextExplainer
from eli5.lime.samplers import MaskingTextSampler
from sklearn.feature_extraction.text import CountVectorizer
from timeit import default_timer as timer
from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier



def check_bias(x, y):
    return "<BIAS>" not in x


def generate_explanation_single(
    goal,
    document,
    categories,
    random_seed=12,    
    custom_n_samples=1500,
    return_explainer=False    
):
    if isinstance(document, str):
        document = textattack.shared.attacked_text.AttackedText(document)

    explainer = TextExplainer(
        clf=LogisticRegression(
            class_weight="balanced",
            random_state=random_seed,
            max_iter=300,
            n_jobs=-1,
        ),
        vec=CountVectorizer(stop_words="english", lowercase=True),
        n_samples=custom_n_samples,
        random_state=random_seed,
        sampler=MaskingTextSampler(random_state=random_seed),
    )

    prediction = categories[goal.get_output(document)]    
    probability = goal.pred_proba(document)    
    probability = float(probability.max())
        
    explainer.fit(document.text, goal.pred_proba_LIME_Sampler)
    

    explanation = explainer.explain_prediction(
        target_names=categories,
        feature_names=explainer.vec_.get_feature_names_out(),
        feature_filter=check_bias,
    )
    if return_explainer:
        return explainer, explanation, prediction, probability
    

    return (explanation, prediction, probability)
