from textattack import Attack
from textattack.attack_recipes import AttackRecipe
from textattack.transformations import WordSwapEmbedding
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.transformations import WordSwapEmbedding
from textattack.search_methods import GreedyWordSwapWIR
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.pre_transformation.max_modification_rate import (
    MaxModificationRate,
)

from src.utils.constants import STOP_WORDS
from src.goals.goal_function import ADV_XAI_GF
from src.search_methods.greedy_word_swap_wir_xai import GreedyWordSwapWIR_XAI

class ADV_XAI_Attack(AttackRecipe):
    @staticmethod
    def build(
        model_wrapper,
        categories, 
        max_candidates=10, 
        modify_rate=0.2, 
        top_n_features=1,
        greedy_search=False):
        # Goal function
        goal_function = ADV_XAI_GF(model_wrapper, categories, top_n_features=top_n_features)
        # Constraints
        constraints = [
            RepeatModification(), # No repeated modification of already modified words
            StopwordModification(stopwords=STOP_WORDS),
            MaxModificationRate(modify_rate, min_threshold=3) #Minium of 3 words are perturbed in a sentence
        ]

        # Same constraints as the TextFooler code but with a min_cos_sim of 0.5 instead of 0.7.
        # (The TextFooler paper claims 0.7, but analysis of the released code and some empirical
        # results show that it's 0.5.)
        constraints.append(WordEmbeddingDistance(min_cos_sim=0.5))

        # Only replace words with the same part of speech (or nouns with verbs)
        constraints.append(PartOfSpeech(allow_verb_noun_swap=True))

        # Universal Sentence Encoder with a minimum angular similarity of ε = 0.5.
        #
        # In the TextFooler code, they forget to divide the angle between the two
        # embeddings by pi. So if the original threshold was that 1 - sim >= 0.5, the
        # new threshold is 1 - (0.5) / pi = 0.840845057
        constraints.append(UniversalSentenceEncoder(
            threshold=0.840845057,
            metric="angular",
            compare_against_original=False,
            window_size=15,
            skip_text_shorter_than_window=True,
        ))

        # Transformation
        transformation = WordSwapEmbedding(max_candidates=max_candidates)

        # Search method
        search_method = GreedyWordSwapWIR_XAI(wir_method="delete")

        return Attack(goal_function, constraints, transformation, search_method)