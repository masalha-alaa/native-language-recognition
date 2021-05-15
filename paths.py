"""
Common paths
"""

import os
from pathlib import Path

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

DB_DIR = ROOT_DIR / "dataset 2021-05-13 11-54-00"
CHUNKS_DIR = DB_DIR / "chunks_2000"
RAW_DATA_DIR = DB_DIR / "raw"
CLEAN_DATA_DIR = DB_DIR / "clean"
SENTENCES_DIR = DB_DIR / "sentences"
TOKENS_DIR = DB_DIR / "tokens"
FEATURES_DIR = DB_DIR / 'features'

IMAGES_DIR = ROOT_DIR / "images"
