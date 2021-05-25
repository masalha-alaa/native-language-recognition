"""
Data visualization in graphs etc.
"""

import matplotlib.pyplot as plt
import pandas as pd
from paths import *
import os
import re
import numpy as np
from pickle import load
from typing import List
from sklearn.feature_extraction.text import CountVectorizer
from src.model_configuration import SetupClass  # just for type hinting
import plotly.figure_factory as ff
from datetime import datetime
from helpers import sort_a_by_b


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


def _scale(column, new_range):
    old_range = (column.min(), column.max())
    return (((column - old_range[0]) * (new_range[1] - new_range[0])) / (old_range[1] - old_range[0])) + new_range[0]


def viz_occurrences_hm(best_k_features, classes, data_setups: List[SetupClass], outputdir, show=True, save=False):
    features_df = pd.DataFrame(dtype=int)
    for data_setup in data_setups:
        # get features specific to this vectorizer
        vocabulary = sorted(set(best_k_features) & set(data_setup.get_features()))
        vectorizer = CountVectorizer(vocabulary=vocabulary, ngram_range=data_setup.ngrams)
        features_df = pd.concat([features_df,
                                 pd.DataFrame(vectorizer.fit_transform(data_setup.x_data).toarray(),
                                              columns=vocabulary, dtype=int)],
                                axis=1)
    features_df.index = classes.values

    # prepare heat map
    hm = features_df.groupby(features_df.index).sum()
    hm = hm.reindex(sort_a_by_b(hm.index, COUNTRIES_FAM_ABC_ORDER))
    hm_display = ((hm / hm.sum()).fillna(0) * 100).round(1)
    # display it
    fig = ff.create_annotated_heatmap(z=(hm.apply(lambda col: _scale(col, (0, 1)))).values,
                                      annotation_text=hm_display.values,
                                      text=hm_display.values,
                                      x=hm.columns.to_list(),
                                      y=hm.index.to_list(),
                                      hoverinfo='x+y+text',
                                      colorscale='Blues')
    fig['layout']['yaxis']['autorange'] = "reversed"
    fig.update_layout(title=dict(text='Heatmap (%)', y=0.03, x=0.5, xanchor='center', yanchor='bottom'),
                      title_font=dict(size=24),
                      xaxis_title='Best Features',
                      yaxis_title='Classes')
    if save:
        fig.write_html(str(outputdir / f'{datetime.now().strftime(DATE_STR_SHORT)} hm.html'))
    if show:
        fig.show()

    return fig


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
