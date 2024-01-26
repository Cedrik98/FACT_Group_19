from textattack import Attack
from textattack.attack_recipes import AttackRecipe
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.pre_transformation.max_modification_rate import (
    MaxModificationRate,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.transformations import WordSwapEmbedding

from src.goals.goal_function import ADV_XAI_GF
from src.goals.initialgoal import initialIndexGF
from src.search_methods.search_random import GreedyWordSwapWIR_RANDOM


class RandomBaseline(AttackRecipe):
    @staticmethod
    def build(
        model_wrapper,
        categories,
        featureSelector=1,
        probThreshold=0.5,
        minWeightDistance=0,
        limeSamples=5000,
        random_seed=1,
        reverse_search_indices=True,
        query_budget=1000,
        success_threshold=0.5,
        model_batch_size=512,
        max_candidates=10,
        modification_rate=0.1,
        logger=None,
        similarity_measure="rbo",
        rbo_p=0.8,
        greedy_search=False,
        search_method="xaifooler",
        crossover="uniform",
        parent_selection="roulette",
        **kwargs,
    ):
        # Swap words with their 10 closest embedding nearest-neighbors.
        # Embedding: Counter-fitted PARAGRAM-SL999 vectors.
        transformation = WordSwapEmbedding(max_candidates=max_candidates)

        # Don't modify the same word twice or the stopwords defined
        # in the TextFooler public implementation.

        # fmt: off
        stopwords = set(
            [
                "a", "about", "above", "across", "after",
                "afterwards", "again", "against", "ain", "all", "almost",
                "alone", "along", "already", "also", "although", "am",
                "among", "amongst", "an", "and", "another", "any", "anyhow",
                "anyone", "anything", "anyway", "anywhere", "are", "aren",
                "aren't", "around", "as", "at", "back", "been", "before",
                "beforehand", "behind", "being", "below", "beside", "besides",
                "between", "beyond", "both", "but", "by", "can", "cannot",
                "could", "couldn", "couldn't", "d", "didn", "didn't", "doesn",
                "doesn't", "don", "don't", "down", "due", "during", "either",
                "else", "elsewhere", "empty", "enough", "even", "ever",
                "everyone", "everything", "everywhere", "except", "first",
                "for", "former", "formerly", "from", "hadn", "hadn't",
                "hasn", "hasn't", "haven", "haven't", "he", "hence", "her",
                "here", "hereafter", "hereby", "herein", "hereupon", "hers",
                "herself", "him", "himself", "his", "how", "however",
                "hundred", "i", "if", "in", "indeed", "into", "is", "isn",
                "isn't", "it", "it's", "its", "itself", "just", "latter",
                "latterly", "least", "ll", "may", "me", "meanwhile", "mightn",
                "mightn't", "mine", "more", "moreover", "most", "mostly",
                "must", "mustn", "mustn't", "my", "myself", "namely", "needn",
                "needn't", "neither", "never", "nevertheless", "next", "no",
                "nobody", "none", "noone", "nor", "not", "nothing", "now",
                "nowhere", "o", "of", "off", "on", "once", "one", "only",
                "onto", "or", "other", "others", "otherwise", "our", "ours",
                "ourselves", "out", "over", "per", "please", "s", "same",
                "shan", "shan't", "she", "she's", "should've", "shouldn",
                "shouldn't", "somehow", "something", "sometime", "somewhere",
                "such", "t", "than", "that", "that'll", "the", "their",
                "theirs", "them", "themselves", "then", "thence", "there",
                "thereafter", "thereby", "therefore", "therein", "thereupon",
                "these", "they", "this", "those", "through", "throughout",
                "thru", "thus", "to", "too", "toward", "towards", "under",
                "unless", "until", "up", "upon", "used", "ve", "was", "wasn",
                "wasn't", "we", "were", "weren", "weren't", "what",
                "whatever", "when", "whence", "whenever", "where",
                "whereafter", "whereas", "whereby", "wherein", "whereupon",
                "wherever", "whether", "which", "while", "whither", "who",
                "whoever", "whole", "whom", "whose", "why", "with", "within",
                "without", "won", "won't", "would", "wouldn", "wouldn't",
                "y", "yet", "you", "you'd", "you'll", "you're", "you've",
                "your", "yours", "yourself", "yourselves"
            ]
        )

        # fmt: on
        constraints = [
            RepeatModification(),
            StopwordModification(stopwords=stopwords),
            MaxModificationRate(modification_rate, 3),
        ]

        # During entailment, we should only edit the hypothesis - keep the premise
        # the same.
        #
        # input_column_modification = InputColumnModification(
        #     ["premise", "hypothesis"], {"premise"}
        # )
        # constraints.append(input_column_modification)

        # Minimum word embedding cosine similarity of 0.5.
        # (The paper claims 0.7, but analysis of the released code and some empirical
        # results show that it's 0.5.)
        constraints.append(WordEmbeddingDistance(min_cos_sim=0.5))

        # Only replace words with the same part of speech (or nouns with verbs)
        constraints.append(PartOfSpeech(allow_verb_noun_swap=True))

        # Universal Sentence Encoder with a minimum angular similarity of ε = 0.5.
        #
        # In the TextFooler code, they forget to divide the angle between the two
        # embeddings by pi. So if the original threshold was that 1 - sim >= 0.5, the
        # new threshold is 1 - (0.5) / pi = 0.840845057
        use_constraint = UniversalSentenceEncoder(
            threshold=0.840845057,
            metric="angular",
            compare_against_original=False,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
        constraints.append(use_constraint)

        # RBO based goal function
        goal_function = ADV_XAI_GF(
            model_wrapper,
            categories=categories,
            featureSelector=featureSelector,
            limeSamples=limeSamples,
            random_seed=random_seed,
            use_cache=True,
            model_batch_size=model_batch_size,
            query_budget=query_budget,
            success_threshold=success_threshold,
            logger=logger,
            similarity_measure=similarity_measure,
            p_RBO=rbo_p,
        )

        # This goal function is used for the initial ranking of which indicies to perturb first.
        indexGoalFunction = initialIndexGF(
            model_wrapper,
            categories=categories,
            featureSelector=featureSelector,
            limeSamples=limeSamples,
            random_seed=random_seed,
            model_batch_size=model_batch_size,
        )

        # TODO: Look why this is different
        search_method = GreedyWordSwapWIR_RANDOM(
            wir_method="random",
            initialIndexGF=indexGoalFunction,
            reverseIndices=False,
            greedy_search=greedy_search,
        )

        return Attack(goal_function, constraints, transformation, search_method)
