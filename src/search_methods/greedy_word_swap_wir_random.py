"""
Greedy Word Swap with Word Importance Ranking
===================================================


When WIR method is set to ``unk``, this is a reimplementation of the search
method from the paper: Is BERT Really Robust?

A Strong Baseline for Natural Language Attack on Text Classification and
Entailment by Jin et. al, 2019. See https://arxiv.org/abs/1907.11932 and
https://github.com/jind11/TextFooler.
"""

import numpy as np

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod
from textattack.shared.validators import (
    transformation_consists_of_word_swaps_and_deletions,
)


class GreedyWordSwapWIR_RANDOM(SearchMethod):
    """An attack that greedily chooses from a list of possible perturbations in
    order of index, after ranking indices by importance.

    Args:
        wir_method: method for ranking most important words
        model_wrapper: model wrapper used for gradient-based ranking
    """

    def __init__(
        self, 
        wir_method="unk", 
        unk_token="[UNK]",
        greedy_search=False):
        self.wir_method = wir_method
        self.unk_token = unk_token
        self.greedy_search = greedy_search

    def _get_index_order(self, initial_text, max_len=-1):
        """Returns word indices of ``initial_text`` in descending order of
        importance."""

        len_text, indices_to_order = self.get_indices_to_order(initial_text)
        if self.wir_method == "random":
            index_order = indices_to_order
            np.random.shuffle(index_order)
            search_over = False
        else:
            raise ValueError(f"Unsupported WIR method {self.wir_method}")

        return index_order, search_over

    def perform_search(self, initial_result):
        attacked_text = initial_result.attacked_text

        # Sort words by order of importance
        index_order, search_over = self._get_index_order(attacked_text)
        i = 0
        cur_result = initial_result
        results = None
        while i < len(index_order) and not search_over:           
            to_modify_word = cur_result.attacked_text.words[index_order[i]]
            if to_modify_word.lower() in self.goal_function.top_features["feature"].values:
                i += 1
                continue

            transformed_text_candidates = self.get_transformations(
                cur_result.attacked_text,
                original_text=initial_result.attacked_text,
                indices_to_modify=[index_order[i]],
            )           
            
            i += 1
            if len(transformed_text_candidates) == 0:
                continue

            # RANDOMLY SELECT ONE OF THE TRANSFORMATION
            transformed_text_candidates = [
                transformed_text_candidates[
                    np.random.choice(range(len(transformed_text_candidates)))
                ]
            ]

            results, search_over = self.get_goal_results(transformed_text_candidates)
            
            results = sorted(results, key=lambda x: -x.score)
            if self.greedy_search:
                # Skip swaps which don't improve the score
                if results[0].score > cur_result.score:
                    cur_result = results[0]
                else:
                    continue
                # If we succeeded, return the index with best similarity.
                if cur_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                    best_result = cur_result
                    # @TODO: Use vectorwise operations
                    max_similarity = -float("inf")
                    for result in results:
                        if result.goal_status != GoalFunctionResultStatus.SUCCEEDED:
                            break
                        candidate = result.attacked_text
                        try:
                            similarity_score = candidate.attack_attrs["similarity_score"]
                        except KeyError:
                            # If the attack was run without any similarity metrics,
                            # candidates won't have a similarity score. In this
                            # case, break and return the candidate that changed
                            # the original score the most.                            
                            break
                        if similarity_score > max_similarity:
                            max_similarity = similarity_score
                            best_result = result
                    return best_result
            else:
                cur_result = results[0]

        return cur_result

    def check_transformation_compatibility(self, transformation):
        """Since it ranks words by their importance, GreedyWordSwapWIR is
        limited to word swap and deletion transformations."""
        return transformation_consists_of_word_swaps_and_deletions(transformation)

    @property
    def is_black_box(self):
        if self.wir_method == "gradient":
            return False
        else:
            return True

    def extra_repr_keys(self):
        return ["wir_method"]
