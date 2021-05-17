from paths import *
from enum import Enum
import random
import os
from pickle import load
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


def sample_random_chunks(filepath, max_chunks=-1):
    chunks = []
    with open(filepath, mode='rb') as f:
        chunks = load(f)
        chunks = random.sample(chunks,
                               len(chunks) if max_chunks == -1
                               else max_chunks if max_chunks < len(chunks)
                               else len(chunks))
    return [random.sample(chunk, len(chunk)) for chunk in chunks]


def get_native_non_native_classes(input_path, max_chunks_per_non_native_country=50, max_chunks_per_class=500):
    native_chunks = []
    non_native_chunks = []
    for chunks_file in os.listdir(input_path):
        if chunks_file.endswith(PKL_LST_EXT):
            is_native = chunks_file.replace(PKL_LST_EXT, '') in NATIVE_COUNTRIES

            chunks = sample_random_chunks(input_path / chunks_file,
                                          -1 if is_native else max_chunks_per_non_native_country)
            chunks = [''.join(chunk) for chunk in chunks]  # convert each chunk to one long string.

            if is_native:
                native_chunks.extend(chunks)
            else:
                non_native_chunks.extend(chunks)

    chunks_per_class = min(max_chunks_per_class, len(native_chunks), len(non_native_chunks))
    if len(native_chunks) > chunks_per_class:
        native_chunks = random.sample(native_chunks, chunks_per_class)
    if len(non_native_chunks) > chunks_per_class:
        non_native_chunks = random.sample(non_native_chunks, chunks_per_class)

    return native_chunks, non_native_chunks


def refine_vocabulary(vocabulary):
    """
    Extract words from (word, count) list.
    """
    if isinstance(vocabulary[0], tuple):
        if isinstance(vocabulary[0][0], tuple):
            return [' '.join(pair[0]) for pair in vocabulary]
        else:
            return [pair[0] for pair in vocabulary]
    else:
        return vocabulary


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.tick_params(labelright=True)
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Drawing normalized confusion matrix...")
    else:
        print('Drawing confusion matrix, without normalization...')

    # thresh = cm.max() / 2.
    thresh = 300
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 ha="center",
                 va='center',
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def run_classifiers(dataframe=None, X=None, y=None, k_fold=10, plot_cm=True):
    """
    Runs multiple classifiers on data in dataFrame (or in X/Y if given) and prints the results.
    Classifiers: SVM , LR, NB , Decision Tree , KNN
    :param dataframe: DataFrame of the data (ignored if X and y are given)
    :param X: Data (dataframe is ignored if this is given)
    :param y: Labels (dataframe is ignored if this is given)
    :param k_fold: K Fold for cross validation.
    :param plot_cm: Whether to draw a confusion matrix or not.
    :return: None
    """

    # Prepare data and targets
    if X is None and y is None:
        X = dataframe.iloc[:, :len(dataframe.columns) - 1].copy().values
        y = dataframe.iloc[-1].values

    classifiers = [(SVC(), False),
                   (LogisticRegression(max_iter=400), True),
                   (MultinomialNB(), False),
                   (RandomForestClassifier(), False),
                   (KNeighborsClassifier(), False)]
    for clf, run in classifiers:
        if run:
            print(clf.__class__.__name__)
            scores = classify(clf, k_fold, X, y, plot_cm)
            print(scores)
            print(f'avg: {round(np.mean(scores) * 100, 2)}%\n')


def classify(clf, K, X, y, plot_cm):
    """
    Classifies data X, and returns the results
    :param clf: The classifier ( for example sk.svm.SVC() )
    :param K: K-Fold for cross validation
    :param X: The data
    :param y: The targets
    :param plot_cm: Whether to draw a confusion matrix or not.
    :return: List of K scores after cross validation
    """

    kfold = KFold(n_splits=K)

    scores = []
    all_actual = []
    all_predicted = []

    for train, test in kfold.split(X):
        # Train and validate (K-Fold cross validation)
        clf.fit(X[train], y[train])
        predicted = clf.predict(X[test])
        actual = y[test]
        scores.extend([accuracy_score(actual, predicted)])
        # gather all predicted and actual for building confusion matrix
        if plot_cm:
            all_predicted.extend(predicted)
            all_actual.extend(actual)
    if plot_cm:
        print('Plotting confusion matrix...')
        # Compute total confusion matrix
        cnf_matrix = confusion_matrix(all_actual, all_predicted)
        np.set_printoptions(precision=2)
        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix,
                              classes=sorted([NATIVE, NON_NATIVE]),
                              title='Confusion matrix')

        # Plot normalized confusion matrix
        # plt.figure()
        # plot_confusion_matrix(cnf_matrix, classes=['Native', 'Non-Native'], normalize=True,
        #                       title='Normalized confusion matrix')

        print('-> Exit plot window to continue...')
        plt.show()

    return scores


class FeatureVector(Enum):
    ONE_K_WORDS = 1
    ONE_K_POS_TRI = 2
    FUNCTION_WORDS = 3


if __name__ == '__main__':
    NATIVE = 1
    NON_NATIVE = 0
    binary = True

    # feature_vector = FeatureVector.ONE_K_WORDS
    # feature_vector = FeatureVector.ONE_K_POS_TRI
    feature_vector = FeatureVector.FUNCTION_WORDS

    if feature_vector == FeatureVector.ONE_K_WORDS:
        feature_vector_path = FeatureVectors.ONE_THOUSAND_WORDS
        chunks_dir = TOKEN_CHUNKS_DIR
        NGRAMS = (1, 1)
    elif feature_vector == FeatureVector.ONE_K_POS_TRI:
        feature_vector_path = FeatureVectors.ONE_THOUSAND_POS_TRI
        chunks_dir = POS_CHUNKS_DIR
        NGRAMS = (3, 3)
    else:  # FUNCTION_WORDS
        feature_vector_path = FeatureVectors.FUNCTION_WORDS
        chunks_dir = TOKEN_CHUNKS_DIR
        NGRAMS = (1, 1)

    ts = datetime.now()
    print('Program started')
    print(ts)

    random.seed(42)

    native_chunks, non_native_chunks = get_native_non_native_classes(chunks_dir)
    df = pd.DataFrame(data=native_chunks + non_native_chunks, columns=['chunks'])
    df['label'] = [NATIVE] * len(native_chunks) + [NON_NATIVE] * len(non_native_chunks)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    with open(FEATURES_DIR / feature_vector_path, mode='rb') as f:
        vocabulary = refine_vocabulary(load(f))

    count_vec = CountVectorizer(vocabulary=vocabulary, ngram_range=NGRAMS, binary=binary)
    ftr_table = count_vec.fit_transform(df['chunks'])

    run_classifiers(X=ftr_table, y=df['label'].values)

    print(datetime.now() - ts)
