"""
Clean the data
"""
from pathlib import Path
from paths import ROOT_DIR
import os
import emoji
import string
from datetime import datetime
import re
import nltk


def clean(line):
    def remove_emojis(text):
        return emoji.get_emoji_regexp().sub(r'', text)

    line = re.sub(r'\(?http\S+\)?', '', remove_emojis(line)).strip()
    if all([ch in string.punctuation for ch in list(line)]):
        line = ''
    return line.replace('[removed]', '').replace('[deleted]', '')


def clean_data(input_dir, output_dir):
    for i, filename in enumerate(os.listdir(input_dir)):
        if filename.endswith('.txt'):
            print(f'{i + 1}. {filename}')
            with open(input_dir / filename, mode='r', encoding='utf-8') as fr:
                with open(output_dir / filename, mode='w', encoding='utf-8') as fw:
                    for line in fr.readlines():
                        clean_line = clean(line)
                        if clean_line:
                            fw.write(f'{clean_line}\n')


def sentecize_data(input_dir, output_dir):
    for i, filename in enumerate(os.listdir(input_dir)):
        if filename.endswith('.txt'):
            print(f'{i + 1}. {filename}')
            with open(input_dir / filename, mode='r', encoding='utf-8') as fr:
                with open(output_dir / filename, mode='w', encoding='utf-8') as fw:
                    for line in fr.readlines():
                        for sentence in re.split(r"(?<=[A-Za-z][A-Za-z])[.?\n!]+|(?<=[0-9)\}\]])[.?\n!]+", line):
                            stripped_sentence = sentence.strip()
                            if len(stripped_sentence) > 2:
                                fw.write(f'{stripped_sentence}\n')


def tokenize_data(input_dir, output_dir):
    for i, filename in enumerate(os.listdir(input_dir)):
        if filename.endswith('.txt'):
            print(f'{i + 1}. {filename}')
            with open(input_dir / filename, mode='r', encoding='utf-8') as fr:
                with open(output_dir / filename, mode='w', encoding='utf-8') as fw:
                    for sentence in fr.readlines():
                        fw.write(f"{' '.join(nltk.word_tokenize(sentence))}\n")


if __name__ == '__main__':
    CLEAN = False
    SENTENCIZE = False
    TOKENIZE = True

    ts = datetime.now()

    input_dir = Path(ROOT_DIR) / "dataset 2021-05-13 11-54-00/raw"
    clean_output_dir = Path(ROOT_DIR) / "dataset 2021-05-13 11-54-00/clean"
    sentences_output_dir = Path(ROOT_DIR) / "dataset 2021-05-13 11-54-00/sentences"
    tokens_output_dir = Path(ROOT_DIR) / "dataset 2021-05-13 11-54-00/tokens"

    if CLEAN:
        print('Cleaning...')
        clean_output_dir.mkdir(exist_ok=True)
        clean_data(input_dir, clean_output_dir)

    if SENTENCIZE:
        print('Splitting text to sentences...')
        sentences_output_dir.mkdir(exist_ok=True)
        sentecize_data(clean_output_dir, sentences_output_dir)

    if TOKENIZE:
        print('Tokenizing...')
        tokens_output_dir.mkdir(exist_ok=True)
        tokenize_data(sentences_output_dir, tokens_output_dir)

    print('')
    print(datetime.now() - ts)
