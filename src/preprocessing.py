"""
Clean the data
"""
from paths import *
from common import *
import os
import emoji
import string
from datetime import datetime
import re
import nltk
from pickle import dump


def clean_line(line):
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
                        clean_line = clean_line(line)
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


def chunkify_data(input_dir, output_dir, minimum_tokens_in_sentence=3, chunk_size=2000):
    for i, filename in enumerate(os.listdir(input_dir)):
        if filename.endswith('.txt'):
            print(f'{i + 1}. {filename}')
            country_chunks_lst = [[]]  # list of country chunks. each inner list is a chunk of size chunk_size.
            current_chunk_size = 0
            with open(input_dir / filename, mode='r', encoding='utf-8') as fr:
                with open(output_dir / filename.replace('.txt', PKL_LST_EXT), mode='wb') as fw:
                    for tokens_line in fr.readlines():
                        tokens_line_split = tokens_line.split()
                        if len(tokens_line_split) >= minimum_tokens_in_sentence:
                            if current_chunk_size >= chunk_size:
                                country_chunks_lst.append([])
                                current_chunk_size = 0
                            country_chunks_lst[-1].append(tokens_line)
                            current_chunk_size += len(tokens_line_split)
                    dump(country_chunks_lst, fw, -1)


if __name__ == '__main__':
    CLEAN = False
    SENTENCIZE = False
    TOKENIZE = False
    CHUNKIFY = True

    ts = datetime.now()

    input_dir = RAW_DATA_DIR
    clean_output_dir = CLEAN_DATA_DIR
    sentences_output_dir = SENTENCES_DIR
    tokens_output_dir = TOKENS_DIR
    chunks_output_dir = CHUNKS_DIR

    if CLEAN:
        print('Cleaning...')
        clean_output_dir.mkdir(exist_ok=True)
        clean_data(input_dir, clean_output_dir)

    if SENTENCIZE:
        print('Sentecising...')
        sentences_output_dir.mkdir(exist_ok=True)
        sentecize_data(clean_output_dir, sentences_output_dir)

    if TOKENIZE:
        print('Tokenizing...')
        tokens_output_dir.mkdir(exist_ok=True)
        tokenize_data(sentences_output_dir, tokens_output_dir)

    if CHUNKIFY:
        print('Chunkifying...')
        chunks_output_dir.mkdir(exist_ok=True)
        chunkify_data(tokens_output_dir, chunks_output_dir)

    print('')
    print(datetime.now() - ts)
