"""
Builds features from data.
"""

from paths import *
from collections import Counter
from datetime import datetime
from pickle import dump
from nltk import ngrams,FreqDist
import plotly.express as px
from shutil import copyfile


def k_most_common(input_dir, k=1000):
    top_k = Counter(''.join(open(input_dir / filename, mode='r', encoding='utf-8').read().lower()
                            for filename in os.listdir(input_dir)).split())
    return top_k.most_common(k)


def k_most_common_trigrams(input_dir, k=1000):
    top_k = FreqDist(ngrams(''.join(open(input_dir / filename, mode='r', encoding='utf-8').read().lower()
                                    for filename in os.listdir(input_dir)).split(), 3))
    return top_k.most_common(k)


def plot_interactive(title, x, y, x_label, y_label, xticks_rot=0, show=True, save=False):
    fig = px.line({x_label: x, y_label: y}, x=x_label, y=y_label, title=title)
    fig.update_layout(hovermode='x')
    fig.update_xaxes(tickangle=xticks_rot)
    if save:
        fig.write_html(str(IMAGES_DIR / f'{len(most_common)} {title.lower()}.html'))
    if show:
        fig.show()
    return fig


if __name__ == '__main__':
    SHOW_GRAPH = False
    SAVE_GRAPH = True
    WORDS = True
    WORDS_NGRAMS = True
    POS = True
    POS_NGRAMS = True

    begin_ts = datetime.now()
    print(begin_ts)

    tokens_input_dir = TOKENS_DIR
    pos_input_dir = POS_DIR
    output_dir = FEATURES_DIR
    output_dir.mkdir(exist_ok=True)

    # function words list is common for all DBs, and is present in the common features dir.

    # build K most common words
    if WORDS:
        words_ts = datetime.now()
        print('K most common words...')
        output_dir.mkdir(exist_ok=True)
        most_common = k_most_common(tokens_input_dir)
        with open(output_dir / f'{len(most_common)} most common words{PKL_LST_EXT}', mode='wb') as f:
            dump(most_common, f, -1)
        print(datetime.now() - words_ts)

        if SHOW_GRAPH or SAVE_GRAPH:
            plot_interactive('Most common words',
                             [pair[0] for pair in most_common],
                             [pair[1] for pair in most_common],
                             'word', 'count', xticks_rot=270,
                             show=SHOW_GRAPH, save=SAVE_GRAPH)

    # build K most common ngrams
    if WORDS_NGRAMS:
        trigrams_ts = datetime.now()
        print('K most common word ngrams (trigrams)...')
        output_dir.mkdir(exist_ok=True)
        most_common = k_most_common_trigrams(tokens_input_dir)
        with open(output_dir / f'{len(most_common)} most common word ngrams{PKL_LST_EXT}', mode='wb') as f:
            dump(most_common, f, -1)
        print(datetime.now() - trigrams_ts)

        if SHOW_GRAPH or SAVE_GRAPH:
            plot_interactive('Most common word ngrams',
                             [' '.join(pair[0]) for pair in most_common],
                             [pair[1] for pair in most_common],
                             'ngram', 'count', xticks_rot=270,
                             show=SHOW_GRAPH, save=SAVE_GRAPH)

    # build K most common POS
    if POS:
        pos_ts = datetime.now()
        print('K most common POS...')
        output_dir.mkdir(exist_ok=True)
        most_common = k_most_common(pos_input_dir)
        with open(output_dir / f'{len(most_common)} most common pos{PKL_LST_EXT}', mode='wb') as f:
            dump(most_common, f, -1)
        print(datetime.now() - pos_ts)

        if SHOW_GRAPH or SAVE_GRAPH:
            plot_interactive('Most common POS',
                             [pair[0] for pair in most_common],
                             [pair[1] for pair in most_common],
                             'pos', 'count', xticks_rot=270,
                             show=SHOW_GRAPH, save=SAVE_GRAPH)

    # build K most common POS ngrams
    if POS_NGRAMS:
        trigrams_ts = datetime.now()
        print('K most common POS ngrams (trigrams)...')
        output_dir.mkdir(exist_ok=True)
        most_common = k_most_common_trigrams(pos_input_dir)
        with open(output_dir / f'{len(most_common)} most common POS trigrams{PKL_LST_EXT}', mode='wb') as f:
            dump(most_common, f, -1)
        print(datetime.now() - trigrams_ts)

        if SHOW_GRAPH or SAVE_GRAPH:
            plot_interactive('Most common POS ngrams',
                             [' '.join(pair[0]) for pair in most_common],
                             [pair[1] for pair in most_common],
                             'ngram', 'count', xticks_rot=270,
                             show=SHOW_GRAPH, save=SAVE_GRAPH)

    print(f'\nTOTAL TIME: {datetime.now() - begin_ts}')
