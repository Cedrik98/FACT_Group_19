
import numpy as np
from textattack.goal_functions.classification.classification_goal_function import (
    ClassificationGoalFunction,
)
from textattack.shared.attacked_text import AttackedText
from src.sim_measures.sim_func import RBO
from src.lime.gen_explanation import generate_explanation_single
from src.utils.format import format_explanation_df

class ADV_XAI_GF(ClassificationGoalFunction):
    def __init__(
        self, 
        model_wrapper,
        categories,
        p_RBO = 0.80,
        top_n_features = 1,
        success_threshold=0.70,
        lime_sr = 1500
        
        
    ):
        super().__init__(model_wrapper)
        self.model = model_wrapper          
        self.categories = categories  
        self.temp_score = None
        self.p_RBO = p_RBO
        self.success_threshold = success_threshold
        self.n_samples = lime_sr
        self.top_n_features = top_n_features

    def init_attack_example(self, attacked_text, ground_truth_output):        
        self.initial_attacked_text = attacked_text
        
        self.ground_truth_output = ground_truth_output
        self.base_explanation = generate_explanation_single(
            self, 
            attacked_text, 
            self.categories, 
            custom_n_samples=self.n_samples, 
            return_explainer=True
        )
        self.base_prediction = self.base_explanation[2]
        self.base_probability = self.base_explanation[3]
        
        self.base_explanation_df = format_explanation_df(
            self.base_explanation[1], target=self.base_prediction 
        )
        
        if self.base_explanation_df.empty:
            # print("Empyt base prediction")
            return False
        # print(self.base_explanation_df)

        self.num_queries = 0        
        self.base_feature_set = set(self.base_explanation_df.get("feature"))
        
        self.top_features = self.base_explanation_df.head(self.top_n_features)
        result = self.get_result(attacked_text, check_skip=True)
        
        return result
    
    def get_result(self, attacked_text, **kwargs):
        """A helper method that queries ``self.get_results`` with a single
        ``AttackedText`` object."""
        results, search_over = self.get_results([attacked_text], **kwargs)
        result = results[0] if len(results) else None
        return result, search_over

    def get_results(self, attacked_text_list, replacement=None, check_skip=False):
        # print("main attack function triggered")
        """For each attacked_text object in attacked_text_list, returns a
        result consisting of whether or not the goal has been achieved, the
        output for display purposes, and a score.

        Additionally returns whether the search is over due to the query
        budget.
        """
        results = []

        self.num_queries += len(attacked_text_list)
        model_outputs = self._call_model(attacked_text_list)

        # print("target_original", target_original)
        for i, (attacked_text, raw_output) in enumerate(
            zip(attacked_text_list, model_outputs)
        ):
            displayed_output = self._get_displayed_output(raw_output)
            goal_status = self._get_goal_status(
                raw_output, attacked_text, check_skip=check_skip
            )
            goal_function_score = self._get_score(raw_output, attacked_text)
            results.append(
                self._goal_function_result_type()(
                    attacked_text,
                    raw_output,
                    displayed_output,
                    goal_status,
                    goal_function_score,
                    self.num_queries,
                    self.ground_truth_output                    
                )
            )
        
        return results, False
    
    def _get_score(self, model_output, attacked_text):
        if self.temp_score is not None:
            return 1 - self.temp_score
        else:
            return 0
    
    def _is_goal_complete(self, model_output, attacked_text):
        self.temp_score = None
        
        
        # Generate explanation
        perturbed_explanation = generate_explanation_single(self, attacked_text, self.categories, custom_n_samples=self.n_samples)
        if self.base_prediction != perturbed_explanation[1]:
            # print("FAILED! Base prediction class must be the same as the attacked prediction class")
            return False
    
        # Set generated explanation as target explanation
        targets = format_explanation_df(perturbed_explanation[0], perturbed_explanation[1])
        if len(targets) == 0:
            # print("FAILED! Explanation prediction does not cover base prediction")

            return False
        

        # check that non of the replacement is within the top-n,
        # except that the replacement is already in the text
        # targets.get('feature')[:self.top_n]        
        if "newly_modified_indices" in attacked_text.attack_attrs:
            if not (attacked_text.attack_attrs["newly_modified_indices"] == set()):                
                new_modified_index = list(
                    attacked_text.attack_attrs["newly_modified_indices"]
                )[0]
                from_w = attacked_text.attack_attrs["previous_attacked_text"].words[
                    new_modified_index
                ]
                to_w = attacked_text.words[new_modified_index]
                

                modified_index = list(attacked_text.attack_attrs["modified_indices"])
                for j in modified_index:
                    to_w_j = attacked_text.words[j]
                    if (
                        to_w_j.lower() not in self.base_feature_set
                        and to_w_j.lower()
                        in targets.get("feature")[: self.top_n_features].values
                    ):
                        
                        return False       
        
        if set(targets['feature'][:self.top_n_features]) == set(self.top_features['feature']):
            # print("FAILED! No feature is has a lower rank")

            return False       
            
        # Compar base explanation with target explanation using RBO
        base_list = self.base_explanation_df.get(
            "feature"
        ).values  # assume that they are already sorted

        target_list = targets.get(
            "feature"
        ).values  # assume that they are already sorted
        
        rboOutput = RBO(target_list, base_list, p=self.p_RBO)
        if rboOutput == False:
            return False
        # print("Internal rboOutput", rboOutput)
        self.temp_score = rboOutput
        if self.temp_score > self.success_threshold:
            # print(
            #     "FAILED! Explanation still too similar, {} {}".format(
            #         self.temp_score, self.success_threshold
            #     )
            # )
                    
            return False
        self.final_explanation = perturbed_explanation        
        return True

    def pred_proba(self, attacked_text):
        """Returns output for display based on the result of calling the
        model."""
        if type(attacked_text) == str:
            attacked_text = AttackedText(attacked_text)

        return np.expand_dims(self._call_model([attacked_text])[0].numpy(), axis=1)
    
    def pred_proba_LIME_Sampler(self, attacked_texts):
        """Returns output for the LIME sampler based on the result of calling the model.
        Expects default LIME sampler output as a tuple of str.
        Use pred_proba for individual str or attacked_text
        """
        # print('pred_proba_LIME_Sampler', attacked_texts)
        # if type(attacked_texts) == str:
        #   attacked_text = AttackedText(attacked_texts)
        
        # output = torch.stack(self._call_model_LIME_Sampler(attacked_texts), 0)
        output = self._call_model_LIME_Sampler(attacked_texts)
        return output.numpy()   
    
    def _call_model_LIME_Sampler(self, attacked_text_list):
        """Gets predictions for a list of ``AttackedText`` objects.

        Gets prediction from cache if possible. If prediction is not in
        the cache, queries model and stores prediction in cache.
        """
                
        if type(attacked_text_list) is tuple:
            attacked_text_list = [
                AttackedText(string)
                for string in attacked_text_list
            ]
        else:
            attacked_text_list = [
                AttackedText(string)
                for string in attacked_text_list[0]
            ]
        
        return self._call_model_uncached(attacked_text_list)

    def generate_explanation(
        self, document, return_explainer=False, custom_n_samples=None, random_seed=12
    ):        
        explainer, explanation, prediction, probability = generate_explanation_single(
            self,
            document,
            self.categories,
            custom_n_samples=custom_n_samples,        
            return_explainer=True,
            random_seed=random_seed
        )
        if return_explainer:
            return explainer

        return explanation, prediction, probability