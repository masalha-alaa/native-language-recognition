"""
Data visualization in graphs etc.
"""

import matplotlib.pyplot as plt
from paths import *
import os
import re
import numpy as np
from pickle import load


def new_figure(title, xlabel, ylabel, x, y):
    fig = plt.gcf()
    plt.title(title)
    plt.xticks(rotation=90)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y')
    plt.bar(x,y)
    fig.set_size_inches((13.5, 4), forward=False)
    return fig


def sentences_count(input_dir, output_dir, show=True, save=False):
    x,y = [],[]
    for country_file in os.listdir(input_dir):
        if country_file.endswith('.txt'):
            x.append(re.sub(r'\..*', '', country_file))
            y.append(sum(1 for _ in open(input_dir / country_file, encoding='utf-8')))
    y,x = zip(*sorted(zip(y,x), reverse=True))
    new_figure('Sentences per Country', 'Country', 'Number of Sentences', x, y)

    if save or show:
        if save:
            plt.savefig(f'{output_dir}/sentences_count.png', bbox_inches='tight', dpi=200)
        if show:
            plt.show()
        plt.close()


def median_sentence_len(input_dir, output_dir, method='average', show=True, save=False):
    def apply_method(lst): return np.mean(lst) if method == 'average' else np.median(lst)

    x,y = [],[]
    for country_file in os.listdir(input_dir):
        if country_file.endswith('.txt'):
            x.append(re.sub(r'\..*', '', country_file))
            y.append(apply_method([len(line.split()) for line in open(input_dir / country_file, encoding='utf-8')]))
    y,x = zip(*sorted(zip(y,x), reverse=True))
    new_figure('Sentences median length per Country', 'Country', 'Median Sentence length (words)', x, y)

    if save or show:
        if save:
            plt.savefig(f'{output_dir}/sentences_length_{method}.png', bbox_inches='tight', dpi=200)
        if show:
            plt.show()
        plt.close()


def chunks_count(input_dir, output_dir, show=True, save=False):
    x,y = [],[]
    for country_file in os.listdir(input_dir):
        if country_file.endswith(PKL_LST_EXT):
            x.append(re.sub(r'\..*', '', country_file))
            y.append(len(load(open(input_dir / country_file, mode='rb'))))
    y,x = zip(*sorted(zip(y,x), reverse=True))
    new_figure('Chunks per Country', 'Country', 'Number of Chunks', x, y)

    if save or show:
        if save:
            plt.savefig(f'{output_dir}/chunks_count.png', bbox_inches='tight', dpi=200)
        if show:
            plt.show()
        plt.close()


if __name__ == '__main__':
    SAVE_IMAGES = True
    SHOW_IMAGES = False

    input_dir = SENTENCES_DIR
    chunks_input_dir = TOKEN_CHUNKS_DIR
    output_dir = IMAGES_DIR
    output_dir.mkdir(exist_ok=True)

    print('Sentences count...')
    sentences_count(input_dir, output_dir, show=SHOW_IMAGES, save=SAVE_IMAGES)

    print('Sentences median length...')
    median_sentence_len(input_dir, output_dir, method='median', show=SHOW_IMAGES, save=SAVE_IMAGES)

    print('Sentences average length...')
    median_sentence_len(input_dir, output_dir, method='average', show=SHOW_IMAGES, save=SAVE_IMAGES)

    print('Chunks count...')
    chunks_count(chunks_input_dir, output_dir, show=SHOW_IMAGES, save=SAVE_IMAGES)