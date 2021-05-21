"""
Picking data from files.
"""


import my_random
from my_random import SEED
import pandas as pd
from collections import defaultdict
from src.model_configuration import *


class ClassesType(Enum):
    BINARY_NATIVITY = 1
    COUNTRY_IDENTIFICATION = 2  # NLI
    LANGUAGE_FAMILY = 3

    def __str__(self):
        return f"{self.__class__.__name__}: {self.name.replace('_', ' ')}"


class DataPicker:
    NATIVE = 'Native'
    NON_NATIVE = 'Non-Native'

    @staticmethod
    def _sample_random_chunks(filepath, min_chunks=0, max_chunks=-1):
        chunks = []
        with open(filepath, mode='rb') as f:
            chunks = load(f)
            if len(chunks) < min_chunks:
                return []
            chunks = my_random.sample(chunks,
                                   len(chunks) if max_chunks == -1
                                   else max_chunks if max_chunks <= len(chunks)
                                   else len(chunks))
        return [my_random.sample(chunk, len(chunk)) for chunk in chunks]

    @staticmethod
    def _get_native_non_native_classes(input_path, max_chunks_per_non_native_country=50, max_chunks_per_class=500):
        min_chunks = min(20, max_chunks_per_non_native_country)

        native_chunks = []
        non_native_chunks = []
        for chunks_file in os.listdir(input_path):
            if chunks_file.endswith(PKL_LST_EXT):
                is_native = chunks_file.replace(PKL_LST_EXT, '') in NATIVE_COUNTRIES

                chunks = DataPicker._sample_random_chunks(input_path / chunks_file, min_chunks,
                                                          -1 if is_native else max_chunks_per_non_native_country)
                chunks = [''.join(chunk) for chunk in chunks]  # convert each chunk to one long string.

                if is_native:
                    native_chunks.extend(chunks)
                else:
                    non_native_chunks.extend(chunks)

        chunks_per_class = min(max_chunks_per_class, len(native_chunks), len(non_native_chunks))
        if len(native_chunks) > chunks_per_class:
            native_chunks = my_random.sample(native_chunks, chunks_per_class)
        if len(non_native_chunks) > chunks_per_class:
            non_native_chunks = my_random.sample(non_native_chunks, chunks_per_class)

        return native_chunks, non_native_chunks

    @staticmethod
    def _get_country_classes(input_path, chunks_per_country=50):
        d = {}
        for chunks_file in os.listdir(input_path):
            if chunks_file.endswith(PKL_LST_EXT):
                chunks = DataPicker._sample_random_chunks(input_path / chunks_file,
                                                          chunks_per_country, chunks_per_country)
                chunks = [''.join(chunk) for chunk in chunks]  # convert each chunk to one long string.
                if chunks:
                    d[chunks_file.replace(PKL_LST_EXT, '')] = chunks
        return d

    @staticmethod
    def _get_family_classes(input_path, max_chunks_per_country=80, chunks_per_fam=300):
        min_chunks = min(20, max_chunks_per_country)

        d = defaultdict(list)  # defaultdict to return an empty list (zero len) if we didn't add the family yet
        for chunks_file in os.listdir(input_path):
            if chunks_file.endswith(PKL_LST_EXT):
                country_name = chunks_file.replace(PKL_LST_EXT, '')
                family = LanguageFamilies.get_country_fam(country_name)
                if family:
                    add_to_fam = min(max_chunks_per_country, chunks_per_fam - len(d[family]))
                    chunks = DataPicker._sample_random_chunks(input_path / chunks_file, min_chunks, add_to_fam)
                    chunks = [''.join(chunk) for chunk in chunks]  # convert each chunk to one long string.
                    if chunks:
                        d[family].extend(chunks)
        return d

    @staticmethod
    def get_data(feature_vector_type: FeatureVectorType, feature_vector_values: FeatureVectorValues,
                 classes_type: ClassesType, chunks_per_class):

        vocabulary_setups = ModelConfiguration.get_configuration(feature_vector_type, feature_vector_values)

        """
        We need to build a separate features table for each vocabulary, and then combine them (stack them horizontally).
        For example, if the selected features were: 1K TOP WORDS, 1K TOP POS-TRI, FUNCTION WORDS, then we would
        need to build:
        TABLE 1 (1K TOP WORDS):
            W1  W2  W3  ...  Wn
        S1
        S2
        S3
        ...
        Sn
        
        TABLE 2 (1K TOP POS-TRI):
            W1  W2  W3  ...  Wn
        S1
        S2
        S3
        ...
        Sn
        
        TABLE 3 (FW):
            W1  W2  W3  ...  Wn
        S1
        S2
        S3
        ...
        Sn
        
        Si = sample, Wi = feature (e.g. a word).
        
        Now in order to fill TABLE 1, we need to sample *random* data from the tokenized DB.
        But in order to fill TABLE 2, we need to sample data from the POS tokenized DB, such that it corresponds
        to TABLE 1 [problem 1].
        Then when filling up TABLE 3, surely we can re-use the data we had sampled for TABLE 1, but for simplicity we
        sample again (of course we need to make sure we re-sample exactly the same data) [problem 2].
        
        [problem 1] is solved by itself, because the way the DBs are built, the order among them is guaranteed.
        To solve [problem 2], we just use the same seed every time we sample, thus guaranteeing to sample the
        same indices every time.
        """

        for vocabulary_setup in vocabulary_setups:
            if classes_type == ClassesType.BINARY_NATIVITY:
                native_chunks, non_native_chunks = DataPicker._get_native_non_native_classes(vocabulary_setup.chunks_dir,
                                                                                             max_chunks_per_class=chunks_per_class)
                df = pd.DataFrame(native_chunks + non_native_chunks, columns=['chunks'])
                df['label'] = [DataPicker.NATIVE] * len(native_chunks) + [DataPicker.NON_NATIVE] * len(non_native_chunks)
                vocabulary_setup.data = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

            elif classes_type == ClassesType.COUNTRY_IDENTIFICATION:
                countries = DataPicker._get_country_classes(vocabulary_setup.chunks_dir, chunks_per_country=chunks_per_class)
                df = pd.DataFrame(sum(countries.values(), []), columns=['chunks'])
                df['label'] = [country for country, chunks in countries.items() for _ in range(len(chunks))]
                vocabulary_setup.data = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

            elif classes_type == ClassesType.LANGUAGE_FAMILY:
                families = DataPicker._get_family_classes(vocabulary_setup.chunks_dir, chunks_per_fam=chunks_per_class)
                df = pd.DataFrame(sum(families.values(), []), columns=['chunks'])
                df['label'] = [family for family, chunks in families.items() for _ in range(len(chunks))]
                vocabulary_setup.data = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

        # The labels are the same for all setups in a particular function call,
        # because 'classes_type' is constant in all iteration.
        # So take any labels column... it doesn't matter.
        return vocabulary_setups, vocabulary_setups[0].data['label']
