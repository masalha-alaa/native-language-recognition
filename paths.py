"""
Common paths.
!!! DO NOT CHANGE THIS FILE'S LOCATION !!!
"""

import os
from pathlib import Path

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

CONTROL_DB = 'dataset control db'
DB_DIR = ROOT_DIR / "dataset 2021-05-16 16-56-59"
# DB_DIR = ROOT_DIR / CONTROL_DB
TOKEN_CHUNKS_DIR = DB_DIR / "token_chunks_2000"
POS_CHUNKS_DIR = DB_DIR / "pos_chunks_2000"
RAW_DIR_NAME = 'raw'
RAW_DATA_DIR = DB_DIR / RAW_DIR_NAME
CLEAN_DATA_DIR = DB_DIR / "clean"
SENTENCES_DIR = DB_DIR / "sentences"
TOKENS_DIR = DB_DIR / "tokens"
POS_DIR = DB_DIR / "pos"
FEATURES_DIR = DB_DIR / 'features'
IMAGES_DIR = DB_DIR / "images"
