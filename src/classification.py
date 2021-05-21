"""
Classification
"""

import argparse
from bisect import bisect_left
from scipy.sparse import hstack, csr_matrix
from src.data_picker import *
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


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
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Drawing normalized confusion matrix...")
    else:
        print('Drawing confusion matrix, without normalization...')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 ha="center",
                 va='center',
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def run_classifiers(dataframe=None, X=None, y=None, k_fold=10, classes=None,
                    verbose=1, results_path=None, plots_path=None,
                    cm_title='Confusion Matrix'):
    """
    Runs multiple classifiers on data in dataFrame (or in X/Y if given) and prints the results.
    Classifiers: SVM , LR, NB , Decision Tree , KNN
    :param dataframe: DataFrame of the data (ignored if X and y are given)
    :param X: Data (dataframe is ignored if this is given)
    :param y: Labels (dataframe is ignored if this is given)
    :param k_fold: K Fold for cross validation.
    :param classes: Classes list for cm. If None, the classes will be inferred from the order in the labels column.
    :param verbose: Debug level
    :param results_path: Path to save results (only if verbose > 2)
    :param plots_path: Path to save plots (only if verbose > 2)
    :param cm_title: Confusion matrix title
    :return: None
    """

    # Prepare data and targets
    if X is None and y is None:
        X = dataframe.iloc[:, :len(dataframe.columns) - 1].copy().values
        y = dataframe.iloc[-1].values

    if classes is None:
        classes = pd.unique(y)

    classifiers = [(SVC(), False),
                   (LogisticRegression(max_iter=500), True),
                   (MultinomialNB(), False),
                   (RandomForestClassifier(), False),
                   (KNeighborsClassifier(), False)]
    scores = {}
    for clf, run in classifiers:
        if run:
            if verbose > 1:
                classification_ts = datetime.now()
                print_log(clf.__class__.__name__, end=' ', file=results_path)

            scores_, all_actual, all_predicted = classify(clf, k_fold, X, y)
            scores_avg = np.mean(scores_)
            scores[clf.__class__.__name__] = scores_

            if verbose > 1:
                print_log(f'({datetime.now() - classification_ts})', file=results_path)

                if verbose > 2:
                    print_log('Plotting confusion matrix...')
                    # Compute total confusion matrix
                    cnf_matrix = confusion_matrix(all_actual, all_predicted, labels=classes)
                    np.set_printoptions(precision=2)
                    # Plot non-normalized confusion matrix
                    fig = plt.figure()
                    plot_confusion_matrix(cnf_matrix,
                                          classes=classes,
                                          title=f'{cm_title}\nAccuracy: {scores_avg*100:.1f}%',
                                          normalize=False)

                    print_log('-> Exit plot window to continue...')

                    if plots_path:
                        # fig.set_size_inches((4, 3), forward=False)
                        fig.set_size_inches((0.22 * len(classes) + 3.8, 0.16 * len(classes) + 2.8), forward=False)
                        plt.savefig(plots_path / f'{datetime.now().strftime(DATE_STR_SHORT)} cm.png',
                                    bbox_inches='tight', dpi=170)
                    plt.show()
                    plt.close()

            if verbose > 0:
                print_log(scores_, file=results_path)
                print_log(f'avg: {round(scores_avg * 100, 2)}%\n', file=results_path)

    return scores


def classify(clf, K, X, y):
    """
    Classifies data X, and returns the results
    :param clf: The classifier ( for example sk.svm.SVC() )
    :param K: K-Fold for cross validation
    :param X: The data
    :param y: The targets
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
        all_predicted.extend(predicted)
        all_actual.extend(actual)

    return scores, all_actual, all_predicted


def sort_a_by_b(a, b):
    res = []
    a_sorted = sorted(a)
    for item in b:
        found = bisect_left(a_sorted, item)
        if found < len(a_sorted) and a_sorted[found] == item:
            del a_sorted[found]
            res.append(item)

    for item in a_sorted:
        res.append(item)

    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", default=3, type=int, help="Verbose level [0-3]")
    args = vars(ap.parse_args())
    verbose = args['verbose']

    print('Program started')

    # Enable / Disable any combination...
    data_types = [
        # ClassesType.BINARY_NATIVITY,
        # ClassesType.COUNTRY_IDENTIFICATION,
        ClassesType.LANGUAGE_FAMILY
    ]

    feature_vector_types = [
        # FeatureVectorType(FtrVectorEnum.ONE_K_WORDS),
        FeatureVectorType(FtrVectorEnum.ONE_K_POS_TRI),
        FeatureVectorType(FtrVectorEnum.FUNCTION_WORDS),
        FeatureVectorType(FtrVectorEnum.ONE_K_POS_TRI) | FeatureVectorType(FtrVectorEnum.FUNCTION_WORDS)
        # # can combine features using the bitwise or (|) operator
    ]

    feature_vector_value_types = [
        FeatureVectorValues.BINARY,
        FeatureVectorValues.FREQUENCY,
        FeatureVectorValues.TFIDF
    ]

    for data_type in data_types:
        for feature_vector_type in feature_vector_types:
            for feature_vector_values in feature_vector_value_types:

                if data_type != ClassesType.BINARY_NATIVITY and feature_vector_values == FeatureVectorValues.BINARY:
                    # This is a bad choice, because the clf will complain about unscaled data.
                    print(f'* SKIPPED SETUP: {data_type}, {feature_vector_values}')
                    continue

                ts = datetime.now()
                print(ts)
                results_path = RESULTS_DIR
                results_path.mkdir(exist_ok=True)
                plots_path = results_path
                results_path /= f'{ts.strftime(DATE_STR_SHORT)} results.txt'

                if data_type == ClassesType.BINARY_NATIVITY:
                    chunks_per_class = 500
                elif data_type == ClassesType.COUNTRY_IDENTIFICATION:
                    chunks_per_class = 50
                elif data_type == ClassesType.LANGUAGE_FAMILY:
                    chunks_per_class = 300
                else:
                    raise NotImplementedError

                config = f'{data_type}\n{feature_vector_type}\n{feature_vector_values}\n{chunks_per_class} chunks per class'
                print(config)
                if verbose > 0 and results_path:
                    with open(results_path, mode='a', encoding='utf8') as f:
                        f.write(f'{config}\n')

                print('Collecting data...', end=' ')
                ts = datetime.now()
                data_setups, labels = DataPicker.get_data(feature_vector_type, feature_vector_values, data_type, chunks_per_class)
                print(f'({datetime.now() - ts})')

                print('Constructing features...', end=' ')
                ts = datetime.now()
                ftr_table = csr_matrix([])
                for i, data_setup in enumerate(data_setups):
                    data_setup.load_vectorizer()
                    if i == 0:
                        # first loop iteration, initiate ftr table
                        ftr_table = data_setup.fit_transform()
                    else:
                        # concatenate ftr tables
                        ftr_table = hstack([ftr_table, data_setup.fit_transform()])
                ftr_table = ftr_table.tocsr()
                print(f'({datetime.now() - ts})')

                print('Classifying...')
                scores_dict = run_classifiers(X=ftr_table, y=labels.values, verbose=verbose,
                                              results_path=results_path,
                                              plots_path=plots_path,
                                              classes=sort_a_by_b(pd.unique(labels), COUNTRIES_ORDER),
                                              cm_title=config)

                print(f'\nTOTAL TIME: {datetime.now() - ts}')
