import textattack
from eli5.lime import TextExplainer
from eli5.lime.samplers import MaskingTextSampler
from sklearn.feature_extraction.text import CountVectorizer
from timeit import default_timer as timer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier #Needed for decisiontrees

def check_bias(x, y):
    return '<BIAS>' not in x

def generate_explanation_single(self, document, custom_n_samples=None, debug=False, return_explainer=False):
    if type(document) == str:
        document = textattack.shared.attacked_text.AttackedText(document)
    # custom_n_samples = 10 
    
    explainer = TextExplainer(
                    clf=LogisticRegression(class_weight='balanced', random_state=self.random_seed, max_iter=300, n_jobs=-1),
                    # clf = CULogisticRegression(class_weight='balanced', max_iter=100, verbose=True),
                    # clf=DecisionTreeClassifier(class_weight='balanced', random_state=self.random_seed, max_depth=10),
                    vec=CountVectorizer(stop_words='english', lowercase=True),
                    n_samples=self.limeSamples if not custom_n_samples else custom_n_samples, 
                    random_state=self.random_seed, 
                    sampler=MaskingTextSampler(random_state=self.random_seed))
    
    prediction = self.categories[self.get_output(document)]
    probability = self.pred_proba(document)
    probability = float(probability.max())
    
    start = timer()   
    explainer.fit(document.text, self.pred_proba_LIME_Sampler)    
    end = timer()

    
    print("Lime took...", end - start)
    
    explanation = explainer.explain_prediction(target_names=self.categories,
                                                feature_names=explainer.vec_.get_feature_names_out(),
                                                feature_filter=check_bias,)
    
    if return_explainer:
        return explainer,explanation,prediction,probability

    return (explanation,prediction,probability)