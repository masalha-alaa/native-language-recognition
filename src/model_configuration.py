"""
Get model configuration according to requested features type.
"""


from enum import Enum
from paths import *
from pickle import load
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class FeatureVectorType(Enum):
    ONE_K_WORDS = 1
    ONE_K_POS_TRI = 2
    FUNCTION_WORDS = 3


class FeatureVectorValues(Enum):
    BINARY = 1
    FREQUENCY = 2
    TFIDF = 3


class ModelConfiguration:
    @staticmethod
    def _refine_vocabulary(vocabulary):
        """
        Extract words from (word, count) list.
        """
        if isinstance(vocabulary[0], tuple):
            if isinstance(vocabulary[0][0], tuple):
                return [' '.join(pair[0]) for pair in vocabulary]
            else:
                return [pair[0] for pair in vocabulary]
        else:
            return vocabulary

    @staticmethod
    def get_configuration(feature_vector_type: FeatureVectorType, feature_vector_vals: FeatureVectorValues):
        if feature_vector_type == FeatureVectorType.ONE_K_WORDS:
            feature_vector_path = FeatureVectorPaths.ONE_THOUSAND_WORDS
            chunks_dir = TOKEN_CHUNKS_DIR
            ngrams = (1, 1)
        elif feature_vector_type == FeatureVectorType.ONE_K_POS_TRI:
            feature_vector_path = FeatureVectorPaths.ONE_THOUSAND_POS_TRI
            chunks_dir = POS_CHUNKS_DIR
            ngrams = (3, 3)
        elif feature_vector_type == FeatureVectorType.FUNCTION_WORDS:
            feature_vector_path = FeatureVectorPaths.FUNCTION_WORDS
            chunks_dir = TOKEN_CHUNKS_DIR
            ngrams = (1, 1)
        else:
            raise ValueError

        with open(FEATURES_DIR / feature_vector_path, mode='rb') as f:
            vocabulary = ModelConfiguration._refine_vocabulary(load(f))

        if feature_vector_vals == FeatureVectorValues.BINARY:
            vectorizer = CountVectorizer(vocabulary=vocabulary, ngram_range=ngrams, binary=True)
        elif feature_vector_vals == FeatureVectorValues.FREQUENCY:
            vectorizer = TfidfVectorizer(vocabulary=vocabulary, ngram_range=ngrams, use_idf=False)
        elif feature_vector_vals == FeatureVectorValues.TFIDF:
            vectorizer = TfidfVectorizer(vocabulary=vocabulary, ngram_range=ngrams, use_idf=True)
        else:
            raise ValueError

        return vectorizer, chunks_dir
