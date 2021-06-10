"""
Classification
"""

import argparse
from src.data_visualization import viz_occurrences_hm
from helpers import *
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
from sklearn.feature_selection import SelectKBest


def select_k_best_2(vocabularies, features_df, labels, k_best):
    """
    Select K best features using sklearn's SelectKBest, which is based on the ANOVA F-value test:
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
    :return: Best K features (unordered)
    """
    best_k_words = SelectKBest(k=k_best)
    best_k_words.fit(features_df, labels)
    best_k_words_idx = best_k_words.get_support(indices=True)
    best_k_words_list = []
    for i in best_k_words_idx:
        for vocabulary in vocabularies:
            for k, v in vocabulary.items():
                if v == i:
                    best_k_words_list.append(k)

    return {'Classifier-Irrelevant': best_k_words_list}


def select_k_best(coefs_dict, feature_names, k_best):
    """
    Select K best features using the classifier's importance ratings.
    :return: Best K features in a descending order.
    """
    # assuming coefficients were scaled.
    # (right now they are scaled in 'run_classifiers()')
    # Source: https://stackoverflow.com/a/34052747/900394
    best_k_features = {}
    for clf_name, coefs in coefs_dict.items():
        importances = coefs
        # partition the importances array, such that the pivot is at the last -k elements
        # (in simpler words: we get the largest argmax k elements at the end)
        k_top_argmax = np.argpartition(importances, -k_best)[-k_best:]
        # sort them in descending order according to their importances
        top_k_sorted = k_top_argmax[np.argsort(importances[k_top_argmax])][::-1]
        best_k_features[clf_name] = feature_names[top_k_sorted].tolist()

    return best_k_features


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
        print('Drawing confusion matrix...')

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
                    scale_coefficients=False,
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
    :param scale_coefficients: Whether to scale classifier's coefficients or not. Should scale if features are not.
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
                   (LogisticRegression(max_iter=1000, random_state=SEED), True),
                   (MultinomialNB(), False),
                   (RandomForestClassifier(), False),
                   (KNeighborsClassifier(), False)]
    scores = {}
    coefficients = {}
    for clf, run in classifiers:
        if run:
            if verbose > 1:
                classification_ts = datetime.now()
                print_log(clf.__class__.__name__, end=' ', file=results_path)

            scores_, all_actual, all_predicted = classify(clf, k_fold, X, y)
            scores_avg = np.mean(scores_)
            scores[clf.__class__.__name__] = scores_
            if scale_coefficients:
                coefficients[clf.__class__.__name__] = np.std(X, axis=0) * clf.coef_[0]
            else:
                coefficients[clf.__class__.__name__] = np.std(X, axis=0)

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
                        fig.set_size_inches((0.22 * len(classes) + 3.8, 0.16 * len(classes) + 3), forward=False)
                        plt.savefig(plots_path / f'{datetime.now().strftime(DATE_STR_SHORT)} cm.png',
                                    bbox_inches='tight', dpi=170)
                    plt.show()
                    plt.close()

            if verbose > 0:
                print_log(scores_, file=results_path)
                print_log(f'avg: {round(scores_avg * 100, 2)}%\n', file=results_path)

    return scores, coefficients


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

    """
    Important note:
    The features in X are filled using CountVectorizer and/or TfidfVectorizer.
    Normally, we should train (fit) the Vectorizers on the training set, and transform them on the test set.
    However, since we didn't initially split the data to train and test sets, but chose to work with cross validation,
    the Vectorizers were trained on the whole data. But even though we're using cross validation, the correct thing
    to do is to train the Vectorizers repetitively in each split on the corresponding training set, and transform
    them (in each split) on the corresponding test set. Otherwise we might suffer from data leakage.
    However, since all our data is from the same domain, and it was shuffled randomly, this won't affect the results
    in our situation. BUT, to use this (trained) model on a new dataset from a different domain, the trained (fitted)
    Vectorizers need to be saved, and then transformed on the new test set.
    This code is not written yet, and thus the model is not a production-ready model.
    """
    for train, test in kfold.split(X):
        # Train and validate (K-Fold cross validation)
        clf.fit(X.iloc[train], y[train])
        predicted = clf.predict(X.iloc[test])
        actual = y[test]
        scores.extend([accuracy_score(actual, predicted)])
        # gather all predicted and actual for building confusion matrix
        all_predicted.extend(predicted)
        all_actual.extend(actual)

    # now fit on all data (for reliably extracting best features later) (also if want to save model...)
    clf.fit(X, y)
    # TODO: Save model

    return scores, all_actual, all_predicted


if __name__ == '__main__':
    SELECT_K_BEST = (50, 0)  # 2 METHODS FOR K BEST (choose k for each. 0 means don't perform selection)

    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", default=3, type=int, help="Verbose level [0-3]")
    args = vars(ap.parse_args())
    verbose = args['verbose']

    print('Program started')

    # Enable / Disable any combination...
    data_types = [
        ClassesType.BINARY_NATIVITY,
        ClassesType.COUNTRY_IDENTIFICATION,
        ClassesType.LANGUAGE_FAMILY
    ]

    feature_vector_types = [
        FeatureVectorType(FtrVectorEnum.ONE_K_WORDS),
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

    # Run over all active configurations and perform classification
    current_results_root = RESULTS_DIR / datetime.now().strftime(DATE_STR_SHORT)
    current_results_root.mkdir(parents=True)
    for data_type in data_types:
        for feature_vector_type in feature_vector_types:
            for feature_vector_values in feature_vector_value_types:

                ts = datetime.now()
                print(ts)
                current_results_file_path = current_results_root / f'{ts.strftime(DATE_STR_SHORT)} results.txt'

                if data_type == ClassesType.BINARY_NATIVITY:
                    chunks_per_class = 500
                elif data_type == ClassesType.COUNTRY_IDENTIFICATION:
                    chunks_per_class = 75
                elif data_type == ClassesType.LANGUAGE_FAMILY:
                    chunks_per_class = 300
                else:
                    raise NotImplementedError

                config = f'{data_type}\n{feature_vector_type}\n{feature_vector_values}\n{chunks_per_class} chunks per class'
                print(config)
                if verbose > 0 and current_results_file_path:
                    with open(current_results_file_path, mode='a', encoding='utf8') as f:
                        f.write(f'{config}\n')

                print('Collecting data...', end=' ')
                ts = datetime.now()
                data_setups, labels = DataPicker.get_data(feature_vector_type, feature_vector_values, data_type, chunks_per_class)
                print(f'({datetime.now() - ts})')

                print('Constructing features...', end=' ')
                ts = datetime.now()
                features_df = pd.DataFrame()
                for data_setup in data_setups:
                    data_setup.load_vectorizer()
                    features_df = pd.concat([features_df,
                                             pd.DataFrame(data_setup.fit_transform().toarray(),
                                                          columns=data_setup.get_features())],
                                            axis=1)
                print(f'({datetime.now() - ts})')

                print('Classifying...')
                scores_dict, coefs_dict = run_classifiers(X=features_df, y=labels.values, verbose=verbose,
                                                          results_path=current_results_file_path,
                                                          plots_path=current_results_root,
                                                          classes=sort_a_by_b(pd.unique(labels), COUNTRIES_FAM_ORDER),
                                                          scale_coefficients=feature_vector_value_types == FeatureVectorValues.BINARY,
                                                          cm_title=config)

                if any(SELECT_K_BEST):
                    print('Getting best features...')

                    best_features_ = [None, None]

                    if SELECT_K_BEST[0]:
                        #  Features are most to least important
                        best_features_[0] = select_k_best(coefs_dict, features_df.columns, SELECT_K_BEST[0])
                    if SELECT_K_BEST[1]:
                        #  Features are NOT most to least important. Order is RANDOM.
                        best_features_[1] = select_k_best_2([data_setup.get_vocabulary() for data_setup in data_setups],
                                                            features_df, labels, SELECT_K_BEST[1])

                    for i, best_features in enumerate(best_features_):
                        if not best_features: continue
                        print(f'SELECT K BEST [METHOD {i+1}]')
                        for clf_name, best_k_ftrs in best_features.items():
                            print_k_best = f"{clf_name} best {SELECT_K_BEST[i]} features:\n" \
                                           f"{', '.join(best_k_ftrs)}\n"
                            print_log(print_k_best, file=current_results_file_path)
                            print('Drawing heatmap...')
                            if data_type != ClassesType.BINARY_NATIVITY:
                                viz_occurrences_hm(best_k_ftrs, labels, data_setups, current_results_root,
                                                   show=False, save=True)

                print(f'Results:\n{current_results_root}')
                print(f'TOTAL TIME: {datetime.now() - ts}\n')
