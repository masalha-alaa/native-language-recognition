"""
Common paths.
!!! DO NOT CHANGE THIS FILE'S LOCATION !!!
"""

import os
from pathlib import Path
from common import *

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

CONTROL_DB = 'dataset control db'
DB_DIR = ROOT_DIR / "dataset 2021-05-16 16-56-59"
# DB_DIR = ROOT_DIR / "dataset 2021-05-23 23-07-00"  # ==> NEW TEST... REMOVING NON ENG WORDS TODO: Try
# DB_DIR = ROOT_DIR / "dataset 2021-05-22 18-05-00"
TOKEN_CHUNKS_DIR = DB_DIR / "token_chunks_2000"
POS_CHUNKS_DIR = DB_DIR / "pos_chunks_2000"
RAW_DIR_NAME = 'raw'
RAW_DATA_DIR = DB_DIR / RAW_DIR_NAME
CLEAN_DATA_DIR = DB_DIR / "clean"
SENTENCES_DIR = DB_DIR / "sentences"
TOKENS_DIR = DB_DIR / "tokens"
POS_DIR = DB_DIR / "pos"
FEATURES_DIR = DB_DIR / 'features'
COMMON_FEATURES_DIR = ROOT_DIR / 'common features'
IMAGES_DIR = DB_DIR / "images"
RESULTS_DIR = DB_DIR / "results"


class FeatureVectorPaths:
    ONE_THOUSAND_WORDS = FEATURES_DIR / f"1000 most common words{PKL_LST_EXT}"
    ONE_THOUSAND_POS_TRI = FEATURES_DIR / f"1000 most common POS trigrams{PKL_LST_EXT}"
    FUNCTION_WORDS = COMMON_FEATURES_DIR / f"en_function_words{PKL_LST_EXT}"
