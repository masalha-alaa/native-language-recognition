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
from multiprocessing import Pool
import random


def clean_a_line(line):
    def remove_emojis(text):
        return emoji.get_emoji_regexp().sub(r'', text)

    line = re.sub(r'\(?http\S+\)?', '', remove_emojis(line)).strip()
    if all([ch in string.punctuation for ch in list(line)]):
        line = ''
    return line.replace('[removed]', '').replace('[deleted]', '')


def clean_data(params):
    input_dir, input_files, output_dir = params
    for filename in input_files:
        if filename.endswith('.txt'):
            print(filename)
            with open(input_dir / filename, mode='r', encoding='utf-8') as fr:
                with open(output_dir / filename, mode='w', encoding='utf-8') as fw:
                    for line in fr.readlines():
                        clean_line = clean_a_line(line)
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


def posify_data(params):
    input_dir, input_files, output_dir = params
    for filename in input_files:
        if filename.endswith('.txt'):
            print(filename)
            with open(output_dir / filename, mode='w', encoding='utf-8') as fw:
                fw.writelines('\n'.join([' '.join([pair[1] for pair in nltk.pos_tag(line.split())])
                                         for line in open(input_dir / filename, mode='r', encoding='utf-8').readlines()]))


def chunkify_data(input_dir, output_dir, minimum_tokens_in_sentence=3, chunk_size=2000):
    for i, filename in enumerate(os.listdir(input_dir)):
        if filename.endswith('.txt'):
            print(f'{i + 1}. {filename}')
            country_chunks_lst = [[]]  # list of country chunks. each inner list is a chunk of size chunk_size.
            current_chunk_size = 0
            with open(input_dir / filename, mode='r', encoding='utf-8') as fr:
                with open(output_dir / filename.replace('.txt', PKL_LST_EXT), mode='wb') as fw:
                    lines = fr.readlines()
                    random.Random(42).shuffle(lines)
                    for tokens_line in lines:
                        tokens_line_split = tokens_line.split()
                        if len(tokens_line_split) >= minimum_tokens_in_sentence:
                            if current_chunk_size >= chunk_size:
                                country_chunks_lst.append([])
                                current_chunk_size = 0
                            country_chunks_lst[-1].append(tokens_line)
                            current_chunk_size += len(tokens_line_split)
                    dump(country_chunks_lst, fw, -1)


if __name__ == '__main__':
    # ESTIMATED TOTAL TIME: 25.5 MINUTES

    CLEAN = True
    SENTENCIZE = True
    TOKENIZE = True
    CHUNKIFY_TOKENS = True
    POSIFY = True
    CHUNKIFY_POS = True

    ts = datetime.now()
    print(ts)

    input_dir = RAW_DATA_DIR
    clean_output_dir = CLEAN_DATA_DIR
    sentences_output_dir = SENTENCES_DIR
    tokens_output_dir = TOKENS_DIR
    tokens_chunks_output_dir = TOKEN_CHUNKS_DIR
    pos_chunks_output_dir = POS_CHUNKS_DIR
    pos_output_dir = POS_DIR

    if CLEAN:
        # EST: 5 minutes
        print('Cleaning...')
        clean_output_dir.mkdir(exist_ok=True)
        raw_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
        print(f'{len(raw_files)} files...')
        pools = 6
        pool = Pool(pools)
        file_groups = [(input_dir_, lst_of_files, output_dir) for input_dir_, output_dir, lst_of_files in
                       zip([input_dir] * pools,
                           [clean_output_dir] * pools,
                           [raw_files[i: i+(len(raw_files)//(pools-1))]
                            for i in range(0, len(raw_files), len(raw_files)//(pools-1))])]
        assert len(file_groups) <= pools
        pool.map(clean_data, file_groups)
        print(f'{datetime.now()}\n')

    if SENTENCIZE:
        # Quick
        print('Sentecizing...')
        sentences_output_dir.mkdir(exist_ok=True)
        sentecize_data(clean_output_dir, sentences_output_dir)
        print(f'{datetime.now()}\n')

    if TOKENIZE:
        # 4 minutes
        print('Tokenizing...')
        tokens_output_dir.mkdir(exist_ok=True)
        tokenize_data(sentences_output_dir, tokens_output_dir)
        print(f'{datetime.now()}\n')

    if CHUNKIFY_TOKENS:
        # Quick
        print('Chunkifying tokens...')
        tokens_chunks_output_dir.mkdir(exist_ok=True)
        chunkify_data(tokens_output_dir, tokens_chunks_output_dir)
        print(f'{datetime.now()}\n')

    if POSIFY:
        # Very slow => On 6 cores it takes 11 minutes.
        print('Posifying ', end='')
        pos_output_dir.mkdir(exist_ok=True)
        tokenized_files = [f for f in os.listdir(tokens_output_dir) if f.endswith('.txt')]
        print(f'{len(tokenized_files)} files...')
        pools = 6
        pool = Pool(pools)
        file_groups = [(input_dir_, lst_of_files, output_dir) for input_dir_, output_dir, lst_of_files in
                       zip([tokens_output_dir] * pools,
                           [pos_output_dir] * pools,
                           [tokenized_files[i: i+(len(tokenized_files)//(pools-1))]
                            for i in range(0, len(tokenized_files), len(tokenized_files)//(pools-1))])]
        assert len(file_groups) <= pools
        pool.map(posify_data, file_groups)
        print(f'{datetime.now()}\n')

    if CHUNKIFY_POS:
        # Quick
        print('Chunkifying POS...')
        pos_chunks_output_dir.mkdir(exist_ok=True)
        chunkify_data(pos_output_dir, pos_chunks_output_dir)
        print(f'{datetime.now()}\n')

    print('')
    print(datetime.now() - ts)
