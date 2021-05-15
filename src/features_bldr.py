"""
Builds features from data.
"""

from paths import *
from common import *
from collections import Counter
from datetime import datetime
from pickle import dump
import matplotlib.pyplot as plt
from nltk import ngrams,FreqDist
import plotly.express as px
import plotly


def k_most_common_words(input_dir, k=1000):
    top_k = Counter(''.join(open(input_dir / filename, mode='r', encoding='utf-8').read()
                                  for filename in os.listdir(input_dir)).split())
    return top_k.most_common(k)


def k_most_common_trigrams(input_dir, k=1000):
    top_k = FreqDist(ngrams(''.join(open(input_dir / filename, mode='r', encoding='utf-8').read()
                                    for filename in os.listdir(input_dir)).split(), 3))
    return top_k.most_common(k)


if __name__ == '__main__':
    SHOW_GRAPH = True
    SAVE_GRAPH = True
    WORDS = True
    NGRAMS = True

    print(datetime.now())

    input_dir = TOKENS_DIR
    output_dir = FEATURES_DIR

    # build K most common words
    if WORDS:
        words_ts = datetime.now()
        print('K most common words...')
        output_dir.mkdir(exist_ok=True)
        most_common = k_most_common_words(input_dir)
        with open(output_dir / f'{len(most_common)} most common words{PKL_LST_EXT}', mode='wb') as f:
            dump(most_common, f, -1)
        print(datetime.now() - words_ts)

        if SHOW_GRAPH or SAVE_GRAPH:
            fig = px.line({'word': [pair[0] for pair in most_common], 'count': [pair[1] for pair in most_common]},
                          x='word', y='count',
                          title='Most Common Words')
            fig.update_layout(hovermode='x')
            fig.update_xaxes(tickangle=270)
            if SAVE_GRAPH:
                fig.write_html(str(IMAGES_DIR / f'{len(most_common)} most common words.html'))
            if SHOW_GRAPH:
                fig.show()

    # build K most common ngrams
    if NGRAMS:
        trigrams_ts = datetime.now()
        print('K most common ngrams (trigrams)...')
        output_dir.mkdir(exist_ok=True)
        most_common = k_most_common_trigrams(input_dir)
        with open(output_dir / f'{len(most_common)} most common trigrams{PKL_LST_EXT}', mode='wb') as f:
            dump(most_common, f, -1)
        print(datetime.now() - trigrams_ts)

        fig = px.line({'ngram': [' '.join(pair[0]) for pair in most_common], 'count': [pair[1] for pair in most_common]},
                      x='ngram', y='count',
                      title='Most Common ngrams')
        fig.update_layout(hovermode='x')
        fig.update_xaxes(tickangle=270)
        if SAVE_GRAPH:
            fig.write_html(str(IMAGES_DIR / f'{len(most_common)} most common ngrams.html'))
        if SHOW_GRAPH:
            fig.show()

    print(datetime.now())
