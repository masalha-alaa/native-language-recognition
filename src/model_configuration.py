"""
Get model configuration according to requested features type.
"""


from enum import Enum
from paths import *
from pickle import load
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class FtrVectorEnum(Enum):
    """
    Enum of feature vector types
    """
    # TODO: Move to FeatureVectorType?
    ONE_K_WORDS = 1 << 0
    ONE_K_POS_TRI = 1 << 1
    FUNCTION_WORDS = 1 << 2


class FeatureVectorType(object):
    """
    Enum-like class of feature vector types
    """
    __secret_key = object()  # a secret key to imitate a private constructor)

    def __init__(self, choice=None, name=None, value=None, secret_key=None):
        if isinstance(choice, FtrVectorEnum):
            self.name = choice.name
            self.value = choice.value
        elif name and value and secret_key == self.__secret_key:
            self.name = name
            self.value = value
        else:
            raise ValueError

    def __str__(self):
        return f"{self.__class__.__name__}: {self.name.replace('_', ' ')}"

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, the_name):
        self._name = the_name

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, the_value):
        self._value = the_value

    def __or__(self, other):
        return FeatureVectorType(choice=None,
                                 name=f'{self.name} & {other.name}',
                                 value=self.value | other.value,
                                 secret_key=self.__secret_key)

    def __and__(self, other):
        return self.value & other.value


class FeatureVectorValues(Enum):
    """
    Enum of feature vector values filling method
    """
    BINARY = 1
    FREQUENCY = 2
    TFIDF = 3

    def __str__(self):
        return f"{self.__class__.__name__}: {self.name.replace('_', ' ')}"


class SetupClass:
    def __init__(self, feature_vector_values_type: FeatureVectorValues, vocabulary, chunks_dir, ngrams):
        self._feature_vector_values_type = feature_vector_values_type
        self._vocabulary = vocabulary
        self.chunks_dir = chunks_dir
        self._ngrams = ngrams
        self._vectorizer = None
        self.data = None
        self.fitted = False

    def load_vectorizer(self):
        """
        Load vocabulary and initialize vectorizer according to the features vector type.
        """
        if self._feature_vector_values_type == FeatureVectorValues.BINARY:
            self._vectorizer = CountVectorizer(vocabulary=self._vocabulary, ngram_range=self._ngrams, binary=True)
        elif self._feature_vector_values_type == FeatureVectorValues.FREQUENCY:
            self._vectorizer = TfidfVectorizer(vocabulary=self._vocabulary, ngram_range=self._ngrams, use_idf=False)
        elif self._feature_vector_values_type == FeatureVectorValues.TFIDF:
            self._vectorizer = TfidfVectorizer(vocabulary=self._vocabulary, ngram_range=self._ngrams, use_idf=True)

    def fit_transform(self):
        """
        Fit and apply vectorizer on data.
        # TODO: Add a boolean "train" option for fitting / fitting & transforming.
        """
        if self._vectorizer is None:
            raise RuntimeError(f'Vectorizer not initialized. Please run {self.load_vectorizer.__name__}() first.')
        # TODO: Save fitted vectorizers.
        self.fitted = True
        return self._vectorizer.fit_transform(self.x_data)

    def get_vocabulary(self):
        """
        :return: Vocabulary if vectorizer is fitted.
        :raises: RuntimeError if vectorizer is not fitted yet.
        """
        if self._vectorizer is not None:
            if self.fitted:
                return self._vectorizer.vocabulary_
            else:
                raise RuntimeError(f'Vectorizer not fitted. Please run {self.fit_transform.__name__}() first.')
        else:
            raise RuntimeError(f'Vectorizer not initialized. Please run {self.load_vectorizer.__name__}() first.')

    @property
    def x_data(self):
        """
        A property to access the data without the labels.
        """
        # TODO: Write this in a better way
        return self.data[self.data.columns[0]]

    @property
    def ngrams(self):
        """
        A property to access the ngrams.
        """
        return self._ngrams

    def get_features(self):
        """
        :return: Vectorizer features in an ascending order.
        """
        if self._vectorizer:
            return sorted(self._vectorizer.vocabulary_.keys(), key=lambda ftr: self._vectorizer.vocabulary_[ftr])

    @property
    def chunks_dir(self):
        """
        A property to access the chunks directory.
        """
        return self._chunks_dir

    @chunks_dir.setter
    def chunks_dir(self, path):
        """
        A property to set the chunks directory.
        """
        self._chunks_dir = path

    @property
    def data(self):
        """
        A property to access the data including the labels.
        """
        return self._data

    @data.setter
    def data(self, the_data):
        """
        A property to set the data (including labels).
        :param the_data: The data to set.
        :type the_data: pd.DataFrame.
        """
        self._data = the_data


class ModelConfiguration:
    @staticmethod
    def _refine_vocabulary(vocabulary):
        """
        Extract words from (word, count) list.
        """
        # TODO: This is done because of the way vocabularies were saved. But in reality it's not needed.
        #  We should ditch the values count from the vocabularies - we don't need them.
        #  Or just save them in metadata files.
        #  After doing that, this whole function can be dismissed.
        if isinstance(vocabulary[0], tuple):
            if isinstance(vocabulary[0][0], tuple):
                return [' '.join(pair[0]) for pair in vocabulary]
            else:
                return [pair[0] for pair in vocabulary]
        else:
            return vocabulary

    @staticmethod
    def get_configuration(feature_vector_type: FeatureVectorType, feature_vector_vals: FeatureVectorValues):
        """
        Return model configuration according to features type.
        """
        vocabularies = []

        if feature_vector_type & FtrVectorEnum.ONE_K_WORDS:
            with open(FEATURES_DIR / FeatureVectorPaths.ONE_THOUSAND_WORDS, mode='rb') as f:
                vocabularies.append(SetupClass(feature_vector_vals,  # values type
                                               ModelConfiguration._refine_vocabulary(load(f)),  # vocabulary
                                               TOKEN_CHUNKS_DIR,  # chunks dir
                                               (1, 1)))  # ngrams

        if feature_vector_type & FtrVectorEnum.ONE_K_POS_TRI:
            with open(FEATURES_DIR / FeatureVectorPaths.ONE_THOUSAND_POS_TRI, mode='rb') as f:
                vocabularies.append(SetupClass(feature_vector_vals,  # values type
                                               ModelConfiguration._refine_vocabulary(load(f)),  # vocabulary
                                               POS_CHUNKS_DIR,  # chunks dir
                                               (3, 3)))  # ngrams

        if feature_vector_type & FtrVectorEnum.FUNCTION_WORDS:
            with open(FEATURES_DIR / FeatureVectorPaths.FUNCTION_WORDS, mode='rb') as f:
                vocabularies.append(SetupClass(feature_vector_vals,  # values type
                                               ModelConfiguration._refine_vocabulary(load(f)),  # vocabulary
                                               TOKEN_CHUNKS_DIR,  # chunks dirr
                                               (1, 1)))  # ngrams

        return vocabularies
