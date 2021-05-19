from common import *
import os
import random
from pickle import load
from enum import Enum
import pandas as pd
from datetime import datetime


class ClassesType(Enum):
    BINARY_NATIVITY = 1
    NATIVE_LANGUAGE_IDENTIFICATION = 2  # NLI
    LANGUAGE_FAMILY = 3


class DataPicker:
    random.seed(42)
    NATIVE = 'Native'
    NON_NATIVE = 'Non-Native'

    @staticmethod
    def _sample_random_chunks(filepath, min_chunks=0, max_chunks=-1):
        chunks = []
        with open(filepath, mode='rb') as f:
            chunks = load(f)
            if len(chunks) < min_chunks:
                return []
            chunks = random.sample(chunks,
                                   len(chunks) if max_chunks == -1
                                   else max_chunks if max_chunks <= len(chunks)
                                   else len(chunks))
        return [random.sample(chunk, len(chunk)) for chunk in chunks]

    @staticmethod
    def _get_native_non_native_classes(input_path, max_chunks_per_non_native_country=50, max_chunks_per_class=500):
        native_chunks = []
        non_native_chunks = []
        for chunks_file in os.listdir(input_path):
            if chunks_file.endswith(PKL_LST_EXT):
                is_native = chunks_file.replace(PKL_LST_EXT, '') in NATIVE_COUNTRIES

                chunks = DataPicker._sample_random_chunks(input_path / chunks_file, 0,
                                                          -1 if is_native else max_chunks_per_non_native_country)
                chunks = [''.join(chunk) for chunk in chunks]  # convert each chunk to one long string.

                if is_native:
                    native_chunks.extend(chunks)
                else:
                    non_native_chunks.extend(chunks)

        chunks_per_class = min(max_chunks_per_class, len(native_chunks), len(non_native_chunks))
        if len(native_chunks) > chunks_per_class:
            native_chunks = random.sample(native_chunks, chunks_per_class)
        if len(non_native_chunks) > chunks_per_class:
            non_native_chunks = random.sample(non_native_chunks, chunks_per_class)

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
    def get_data(classes_type: ClassesType, chunks_dir):
        if classes_type == ClassesType.BINARY_NATIVITY:
            native_chunks, non_native_chunks = DataPicker._get_native_non_native_classes(chunks_dir)
            df = pd.DataFrame(data=native_chunks + non_native_chunks, columns=['chunks'])
            df['label'] = [DataPicker.NATIVE] * len(native_chunks) + [DataPicker.NON_NATIVE] * len(non_native_chunks)
        elif classes_type == ClassesType.NATIVE_LANGUAGE_IDENTIFICATION:
            countries = DataPicker._get_country_classes(chunks_dir)
            df = pd.DataFrame(data=sum(countries.values(), []), columns=['chunks'])
            df['label'] = [country for country, chunks in countries.items() for _ in range(len(chunks))]
        elif classes_type == ClassesType.LANGUAGE_FAMILY:
            raise NotImplementedError
        else:
            raise ValueError

        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        return df
