import scipy 
import numpy
def monkeypath_itemfreq(sampler_indices):
   return zip(*numpy.unique(sampler_indices, return_counts=True))
scipy.stats.itemfreq=monkeypath_itemfreq

import textattack
from textattack.shared.utils import words_from_text
from eli5.lime import TextExplainer
from eli5.lime.samplers import MaskingTextSampler, MaskingTextSamplers  
from sklearn.feature_extraction.text import CountVectorizer
from timeit import default_timer as timer
from datasets import load_dataset, ClassLabel

from sklearn.linear_model import LogisticRegression

import math
import numpy as np

from scipy import stats
from utils.file_create import format_explanation_df

def SM(list1, list2):
    coef, p = stats.spearmanr(list1, list2)
    return 1- max(0, coef)

def p_generator(p,d):
    def sum_series(p, d):
       # tail recursive helper function
       def helper(ret, p, d, i):
           term = math.pow(p, i)/i
           if d == i:
               return ret + term
           return helper(ret + term, p, d, i+1)
       return helper(0, p, d, 1)
    
    return  1 - math.pow(p, d-1) + (((1-p)/p) * d *(np.log(1/(1-p)) - sum_series(p, d-1)))

def find_p(top_n_mass = 0.9):
    # top_n_mass = 0.90 #What percentage of mass we wish to have on the top n features.
    n = 3 #Number of top features
    for i in range(1,100,1):
        p = i/100
        output = p_generator(p,n)
        if abs(output-top_n_mass) < 0.01:
            print("Set rbo_p = ",p, " for ", output*100, "% mass to be assigned to the top ", n, " features." )
            break

    # compute RBO
def RBO(list1,list2,p):
    comparisonLength = min(len(list1),len(list2))
    set1 = set()
    set2 = set()
    summation = 0
    for i in range(comparisonLength):
        set1.add(list1[i])
        set2.add(list2[i])            
        summation += math.pow(p,i+1) * (len(set1&set2) / (i+1))
    return ((len(set(list1)&set(list2))/comparisonLength) * math.pow(p,comparisonLength)) + (((1-p)/p) * summation)


     # compute Center of Mass
def COM(attacked_text,perturbedExplanation):
        attacked_text_list = attacked_text.words
        ordered_weights = [0] * len(attacked_text_list)
        p_df = format_explanation_df(perturbedExplanation[0], target=perturbedExplanation[1])
        features = p_df.get('feature').values.tolist()
        weights = p_df.get('weight').values.tolist()
        #occurences = p_df.get('value').values.tolist()

        feat2weight = dict(zip(features, weights))
        for j in range(len(attacked_text_list)):
            ordered_weights[j] = feat2weight.get(attacked_text_list[j], 0)
        
        total_mass = np.sum(np.abs(ordered_weights))

        center_mass = total_mass / 2
        #print("Center mass = ", center_mass)
              
        current_mass = 0
        for i in range(len(ordered_weights)):
            current_mass += abs(ordered_weights[i])
            #print("Current mass = ", current_mass)
            try:
                if current_mass >= center_mass and ordered_weights[i+1] != 0:
                    return i
            except:
                return i

         # compute Center of Mass weighted by feature index
def COM_rank_weight(attacked_text,perturbedExplanation):
        attacked_text_list = attacked_text.words
        ordered_weights = [0] * len(attacked_text_list)
        p_df = format_explanation_df(perturbedExplanation[0], target=perturbedExplanation[1])
        features = p_df.get('feature').values.tolist()
        weights = p_df.get('weight').values.tolist()
        #occurences = p_df.get('value').values.tolist()

        feat2weight = dict(zip(features, weights))
        for j in range(len(attacked_text_list)):
            ordered_weights[j] = feat2weight.get(attacked_text_list[j], 0)*(j+1)

        #print(attacked_text_list)        
        #print(ordered_weights)
        
        total_mass = np.sum(np.abs(ordered_weights))
            
        center_mass = total_mass / 2
        #print("Center mass = ", center_mass)
              
        current_mass = 0
        for i in range(len(ordered_weights)):
            current_mass += abs(ordered_weights[i])
            #print("Current mass = ", current_mass)
            try:
                if current_mass >= center_mass and ordered_weights[i+1] != 0:
                    return i
            except:
                return i       
    
       # compute proportion of mass associated with the center index
def COM_proportional(attacked_text,perturbedExplanation):
        attacked_text_list = attacked_text.words
        ordered_weights = [0] * len(attacked_text_list)
        p_df = format_explanation_df(perturbedExplanation[0], target=perturbedExplanation[1])
        features = p_df.get('feature').values.tolist()
        weights = p_df.get('weight').values.tolist()
        #occurences = p_df.get('value').values.tolist()

        feat2weight = dict(zip(features, weights))
        for j in range(len(attacked_text_list)):
            ordered_weights[j] = feat2weight.get(attacked_text_list[j], 0)

        #print(attacked_text_list)        
        #print(ordered_weights)
        
        total_mass = np.sum(np.abs(ordered_weights))

        center_mass = total_mass / 2
        #print("Center mass = ", center_mass)
              
        current_mass = 0
        for i in range(len(ordered_weights)):
            current_mass += abs(ordered_weights[i])
            if current_mass >= center_mass and ordered_weights[i+1] != 0:
                    break
    
        return current_mass / total_mass
    
def l2(attacked_text,baseExplanation,perturbedExplanation):
        #t0 = timer()
        attacked_text_list = words_from_text(attacked_text)
        ordered_weights = [0] * len(attacked_text_list)
        p_df = format_explanation_df(perturbedExplanation[0], target=baseExplanation[1])
        features = p_df.get('feature').values.tolist()
        weights = p_df.get('weight').values.tolist()
        # occurences = perturbedExplanation.get('value').values.tolist()
        base_ordered_weights = [0] * len(attacked_text_list)
        baseFeatures = format_explanation_df(baseExplanation[0], baseExplanation[1]).get('feature').values.tolist()
        baseWeights = format_explanation_df(baseExplanation[0], baseExplanation[1]).get('weight').values.tolist()

        feat2weightBase = dict(zip(baseFeatures, baseWeights))
        for j in range(len(attacked_text_list)):
            base_ordered_weights[j] = feat2weightBase.get(attacked_text_list[j], 0)


        baseOrderedWeights = np.array(base_ordered_weights)
        #baseTotalMass = np.sum(np.abs(base_ordered_weights))

        feat2weight = dict(zip(features, weights))
        for j in range(len(attacked_text_list)):
            ordered_weights[j] = feat2weight.get(attacked_text_list[j], 0)

        l2 = np.linalg.norm(baseOrderedWeights - np.array(ordered_weights))
        #t1 = timer()
        #print("l2 took...", t1 - t0)

        return l2
    
def jaccard(list1,list2):
        set1 = set(list1)
        set2 = set(list2)
        
        return (len(set.intersection(set1,set2)) / len(set.union(set1,set2)) )
    
def jaccard_weighted(baseExplanation,perturbedExplanation):
        #Need ordered weights and features from the base explanation
        
        #for each feature in base explanation, if it does not exist in the perturbed explanation reduce 
        # sim by its (normalized) weight in the base explanation
        
        sim = 1
        p_df = format_explanation_df(perturbedExplanation[0], target=baseExplanation[1])
        features = p_df.get('feature').values.tolist()

        exdf = format_explanation_df(baseExplanation[0], baseExplanation[1])
        exdf['weight'] = exdf['weight'].abs()
        #print(exdf)
        exdf = exdf.sort_values(by='weight',ascending=False,ignore_index=True)

        baseAbsWeightSum = sum(exdf['weight'])
        baseOrderedExplanation = exdf
        
        for i in range(len(baseOrderedExplanation)):
            if baseOrderedExplanation['feature'][i] not in features:
                       sim -= baseOrderedExplanation['weight'][i]/baseAbsWeightSum
        
        return sim
            
def kendall(baseExplanation,perturbedExplanation):
        
        p_df = format_explanation_df(perturbedExplanation[0], target=baseExplanation[1])

        p_df['weight'] = p_df['weight'].abs()
        p_df = p_df.sort_values(by='weight',ascending=False)
        
        features = p_df.get('feature').values.tolist()
        current_dissonance = 0


        exdf = format_explanation_df(baseExplanation[0], baseExplanation[1])
        exdf['weight'] = exdf['weight'].abs()
        exdf = exdf.sort_values(by='weight',ascending=False,ignore_index=True)

        baseOrderedExplanation = exdf
        
        l1 = len(baseOrderedExplanation['feature'])
        l2 = len(features)
        
                       
        if l1 > l2:
            max_dissonance = l1
            diff = l1-l2
            shorter_explanation = l2
        else:
            max_dissonance = l2
            diff = l2-l1
            shorter_explanation = l1
                       
        
        for i in range(shorter_explanation):
            if baseOrderedExplanation['feature'][i] != features[i]:
                       current_dissonance += 1
                        
        current_dissonance += diff
        
        return 1 - (current_dissonance / max_dissonance)
    
def kendall_weighted(baseExplanation,perturbedExplanation):
        
        p_df = format_explanation_df(perturbedExplanation[0], target=baseExplanation[1])
        p_df['weight'] = p_df['weight'].abs()
        p_df = p_df.sort_values(by='weight',ascending=False)
        
        features = p_df.get('feature').values.tolist()
        current_dissonance = 0
        dissonant_weights = 0

        exdf = format_explanation_df(baseExplanation[0], baseExplanation[1])
        exdf['weight'] = exdf['weight'].abs()
        exdf = exdf.sort_values(by='weight',ascending=False,ignore_index=True)

        baseAbsWeightSum = sum(exdf['weight'])
        baseOrderedExplanation = exdf
        
        
        l1 = len(baseOrderedExplanation)
        l2 = len(features)
        
                       
        if l1 > l2:
            max_dissonance = l1
            diff = l1-l2
        else:
            max_dissonance = l2
            diff = l2-l1
                       
        for i in range(l1):
            if i > l2-1:
                       current_dissonance += 1
                       dissonant_weights += baseOrderedExplanation['weight'][i]/baseAbsWeightSum
            elif baseOrderedExplanation['feature'][i] != features[i]:
                       current_dissonance += 1
                       dissonant_weights += baseOrderedExplanation['weight'][i]/baseAbsWeightSum

        current_dissonance += diff
        return (1 - (current_dissonance / max_dissonance)) * (1-dissonant_weights)       
        
    
def spearman(baseExplanation,perturbedExplanation):
        
        #l1 distance between features
        
        #for each feature in base explanation, calculate the distance if it remains in the perturbed explanation,
        #otherwise penalize with some value (half of the length of the explanation (uniform mean of possible distance))
        
        #maximum total distance is 1/2 floor(explanation_Size squared)
        
        #sum distances divide by max distance

        p_df = format_explanation_df(perturbedExplanation[0], target=baseExplanation[1])

        p_df['weight'] = p_df['weight'].abs()
        p_df = p_df.sort_values(by='weight',ascending=False)
        
        features = p_df.get('feature').values.tolist()

        exdf = format_explanation_df(baseExplanation[0], baseExplanation[1])
        exdf['weight'] = exdf['weight'].abs()
        exdf = exdf.sort_values(by='weight',ascending=False,ignore_index=True)

        baseOrderedExplanation = exdf
        
        current_distance = 0
        
        l1 = len(baseOrderedExplanation)
        
        max_distance = int((l1*l1) / 2)
        penalty = int(l1 / 2)
                                        
        for i in range(l1):
            if baseOrderedExplanation['feature'][i] in features:
                       current_distance += abs(i - features.index(baseOrderedExplanation['feature'][i]))
            else:
                       current_distance += penalty
                            
        return 1 - (current_distance / max_distance)               
            
def spearman_weighted(baseExplanation,perturbedExplanation):
        
        #for each feature in base explanation, calculate the distance if it remains in the perturbed explanation,
        #otherwise penalize with normalized weight * maximum distance
        
        p_df = format_explanation_df(perturbedExplanation[0], target=baseExplanation[1])

                
        p_df['weight'] = p_df['weight'].abs()
        p_df = p_df.sort_values(by='weight',ascending=False)
        
        features = p_df.get('feature').values.tolist()
        
        exdf = format_explanation_df(baseExplanation[0], baseExplanation[1])
        exdf['weight'] = exdf['weight'].abs()
        exdf = exdf.sort_values(by='weight',ascending=False,ignore_index=True)

        baseAbsWeightSum = sum(exdf['weight'])
        baseOrderedExplanation = exdf
        
        
        
        l1 = len(baseOrderedExplanation)
        
        max_distance = int((l1*l1) / 2)
        current_distance = max_distance
      
        
        missing_features = 0
        missing_feature_weight = 0
                       
        for i in range(l1):
            if baseOrderedExplanation['feature'][i] in features:
                       current_distance -= abs(i - features.index(baseOrderedExplanation['feature'][i]))
            else:
                       missing_features += 1
                       missing_feature_weight += baseOrderedExplanation['weight'][i]/baseAbsWeightSum
                       
                       current_distance -= max_distance * baseOrderedExplanation['weight'][i]/baseAbsWeightSum
        
        
        return (current_distance / max_distance)




def generate_comparative_similarities(attacked_text,baseExplanation,perturbedExplanation,RBO_weights = None):
    sims = []

    if RBO_weights is None:
         RBO_weights = [0.6,0.7,0.8,0.9]
            
    df1 = format_explanation_df(baseExplanation[0], target=baseExplanation[1])
    df2 = format_explanation_df(perturbedExplanation[0], target=baseExplanation[1])

    baseList = df1.get('feature').values
    perturbedList = df2.get('feature').values
    
    
    #Set base Center of mass
    #COM
    #baseCOM = COM(attacked_text,format_explanation_df(baseExplanation[0], baseExplanation[1]))

    #COM_proportional 
    #basePropCOM = COM_proportional(attacked_text,format_explanation_df(self.baseExplanation[0], baseExplanation[1]))

    #COM_rank_weighted
    #baseRwCOM = COM_rank_weight(attacked_text,format_explanation_df(self.baseExplanation[0], baseExplanation[1]))

    for w in RBO_weights:
        sims.append(RBO(perturbedList,baseList,w))
    sims.append(jaccard(perturbedList,baseList))
    sims.append(jaccard_weighted(baseExplanation,perturbedExplanation))
    #sims.append(COM(attacked_text,perturbedExplanation))
    #sims.append(COM_proportional(attacked_text,perturbedExplanation))
    #sims.append(COM_rank_weight(attacked_text,perturbedExplanation))
    sims.append(kendall(baseExplanation,perturbedExplanation))
    sims.append(kendall_weighted(baseExplanation,perturbedExplanation))
    sims.append(spearman(baseExplanation,perturbedExplanation))
    sims.append(spearman_weighted(baseExplanation,perturbedExplanation))
    sims.append(l2(attacked_text,baseExplanation,perturbedExplanation))

    
    return sims

def check_bias(x, y):
    return '<BIAS>' not in x

def generate_explanation_single(self, document, custom_n_samples=None, debug=False, return_explainer=False):
    if type(document) == str:
        document = textattack.shared.attacked_text.AttackedText(document)

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

    if debug:
        print("Lime took...", end - start)

    explanation = explainer.explain_prediction(target_names=self.categories,
                                                feature_names=explainer.vec_.get_feature_names_out(),
                                                feature_filter=check_bias,)
   
    if return_explainer:
        return explainer,explanation,prediction,probability

    return (explanation,prediction,probability)