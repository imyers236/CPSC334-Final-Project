##############################################
# Programmer: Ian Myers
# Class: CPSC 322-02, Fall 2024
# Programming Assignment #7
# 11/11/24
# Description: This program creates evaulators for predictors
##############################################

import numpy as np # use numpy's random number generation
import copy
from tabulate import tabulate
import operator

from mysklearn import myutils

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    # assigns random state
    if random_state is None:
        np.random.seed(0)
    else:
        np.random.seed(random_state)

    # gets the amount of instances from the ratio
    if isinstance(test_size,float):
        test_size = int(np.ceil(test_size * len(y)))

    X_train = copy.deepcopy(X)
    y_train = copy.deepcopy(y)
    # shuffles
    if shuffle:
        myutils.randomize_in_place(X_train, parallel_list=y_train)
    X_test = []
    y_test = []
    # takes values for test
    for index in range(test_size):
        X_test.insert(0,X_train.pop(-1))
        y_test.insert(0,y_train.pop(-1))

    return X_train, X_test, y_train, y_test

def kfold_split(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    # list of indices to pick from
    indices = list(range(0,len(X)))
    # assigns random state
    if random_state is None:
        np.random.seed(0)
    else:
        np.random.seed(random_state)
    # shuffles
    if shuffle:
        myutils.randomize_in_place(indices, parallel_list=None)
    # checks if last fold must be different
    uneven = False
    if len(X) % n_splits != 0:
        uneven = True
        test_size = len(X) // n_splits + 1
        test_size_end = len(X) // n_splits
    else:
        test_size = len(X) // n_splits
    folds = []
    # fills up indices in test order
    for i in range(n_splits):
        if (i+1 == n_splits and uneven):
            test = indices[i*test_size:i*test_size+test_size_end]
            train = [val for val in indices if val not in test]
            folds.append((train,test))
        else:
            test = indices[i*test_size:i*test_size+test_size]
            train = [val for val in indices if val not in test]
            folds.append((train,test))
    return folds

# BONUS function
def stratified_kfold_split(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    # Use a dictionary to divide and conquer then take one from every dict indice
    # list of indices to pick from
    indices = list(range(0,len(X)))
    
    # assigns random state
    if random_state is None:
        np.random.seed(0)
    else:
        np.random.seed(random_state)
    # shuffles
    if shuffle:
        myutils.randomize_in_place(indices, parallel_list=None)
    # checks if last fold must be different
    uneven = False
    if len(X) % n_splits != 0:
        uneven = True
        test_size = len(X) // n_splits + 1
        test_size_end = len(X) // n_splits
    else:
        test_size = len(X) // n_splits
    folds = []
    # fills up indices in test order
    for i in range(n_splits):
        if (i+1 == n_splits and uneven):
            test = indices[i*test_size:i*test_size+test_size_end]
            train = [val for val in indices if val not in test]
            folds.append((train,test))
        else:
            test = indices[i*test_size:i*test_size+test_size]
            train = [val for val in indices if val not in test]
            folds.append((train,test))
    return folds

def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results

    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
        Sample indexes of X with replacement, then build X_sample and X_out_of_bag
            as lists of instances using sampled indexes (use same indexes to build
            y_sample and y_out_of_bag)
    """
    # assigns random state
    if random_state is None:
        np.random.seed(0)
    else:
        np.random.seed(random_state)

    # assigns samples length
    if n_samples is None:
        n_samples = len(X)

    X_sample = []
    sample_indices = np.random.choice(len(X), n_samples, replace=True)
    for i in sample_indices:
        X_sample.append(X[i])
    X_out_of_bag = [val for val in X if val not in X_sample]
    if y is None:
        y_sample = None
        y_out_of_bag = None
    else:
        y_sample = []
        y_out_of_bag = []
        for i in sample_indices:
            y_sample.append(y[i])
        for j in range(len(y)):
            if j not in sample_indices:
                y_out_of_bag.append(y[j])

    return X_sample, X_out_of_bag, y_sample, y_out_of_bag

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    # initializes matrix
    matrix = [[0 for i in range(len(labels))] for j in range(len(labels))]
    for i in range(len(y_true)):
        # finds index for true and predicted
        row = labels.index(y_true[i])
        col = labels.index(y_pred[i])
        # adds to the correct row
        matrix[row][col] = matrix[row][col] + 1
    return matrix

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    correct = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            correct += 1
    if normalize:
        acc = correct / len(y_pred)
        return acc
    else:
        return correct

def error_rate(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        error_rate(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).
    """
    falses = 0
    for i in range(len(y_pred)):
        if y_pred[i] != y_true[i]:
            falses += 1
    if normalize:
        error_rate = falses / len(y_pred)
        return error_rate
    else:
        return error_rate

def random_subsample(clf, X, y, k_sub_samples=10, test_size=0.33, discretizer=None):
    """
    Recieves the classifier, X, y, k amount of subsamples, test size and discretizer.
    Uses train_test_split, fits and predicts, k times and returns the average accuracy and error rate
    """
    acc = []
    error = []
    # repeatedly call train test split and get accuracy and error rate
    for i in range(k_sub_samples):
        random_state = i + np.random.random_integers(0, 50)
        X_train, X_test, y_train, y_test = train_test_split(X, y, 0.33, random_state, True)
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        acc.append(accuracy_score(y_test, y_pred,True))
        error.append(error_rate(y_test, y_pred,True))
    # get avg of acc and error rate
    avg_acc = sum(acc) / len(acc)
    avg_error = sum(error) / len(error)
    return avg_acc, avg_error

def cross_val_predict(clf, X, y, k_sub_samples=10, discretizer=None):
    """
    Recieves the classifier, X, y, k amount of folds, discretizer.
    Uses kfold_split, fits and predicts, k times and returns the average accuracy and error rate
    """
    y_test_total = []
    y_pred_total = []
    random_state = np.random.random_integers(0, 100)
    folds = kfold_split(X, k_sub_samples, random_state, True)
    # repeatedly iterate through each fold and get accuracy and error rate
    for train_indices, test_indices in folds:
        X_train = [X[i] for i in train_indices]
        y_train = [y[i] for i in train_indices]
        X_test = [X[i] for i in test_indices]
        y_test = [y[i] for i in test_indices]
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        y_test_total.extend(y_test)
        y_pred_total.extend(y_pred)
    # get metrics of total lists
    acc = accuracy_score(y_test_total, y_pred_total,True)
    error = error_rate(y_test_total, y_pred_total,True)
    precision = binary_precision_score(y_test_total, y_pred_total)
    recall = binary_recall_score(y_test_total, y_pred_total)
    f1 = binary_f1_score(y_test_total, y_pred_total)
    return acc, error, precision, recall, f1

def bootstrap_method(clf, X, y, k_sub_samples=10, n_samples=None, discretizer=None):
    """
    Recieves the classifier, X, y, k amount of subsamples, test size and discretizer.
    Uses bootstrap sample, fits and predicts, k times and returns the average accuracy and error rate
    """
    acc = []
    error = []
    # repeatedly call bootstrap sample and get accuracy and error rate
    for i in range(k_sub_samples):
        random_state = i + np.random.random_integers(0, 50)
        X_train, X_test, y_train, y_test = bootstrap_sample(X, y, n_samples, random_state)
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        acc.append(accuracy_score(y_test, y_pred,True))
        error.append(error_rate(y_test, y_pred,True))
    # get avg of acc and error rate
    avg_acc = sum(acc) / len(acc)
    avg_error = sum(error) / len(error)
    return avg_acc, avg_error

def pretty_print_confusion_matrix(clf, X, y, labels, title, k_sub_samples=10, discretizer=None):
    header = copy.deepcopy(labels)
    random_state = np.random.random_integers(0, 100)
    folds = kfold_split(X, k_sub_samples, random_state, True)
    y_test_total = []
    y_pred_total = []
    # repeatedly iterate through each fold and get each y_test and y_pred and add it to totals
    for train_indices, test_indices in folds:
        X_train = [X[i] for i in train_indices]
        y_train = [y[i] for i in train_indices]
        X_test = [X[i] for i in test_indices]
        y_test = [y[i] for i in test_indices]
        y_test_total.extend(y_test)
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        y_pred_total.extend(y_pred)
    matrix = confusion_matrix(y_test_total, y_pred_total, header)
    # add header to front of each row, total and recognition to end
    for i,l in enumerate(header):
        matrix[i].append(sum(matrix[i]))
        if sum(matrix[i]) == 0:
            recognition = 0
        else:
            recognition = round(matrix[i][i] / matrix[i][-1] * 100)
        matrix[i].append(recognition)
        matrix[i].insert(0,l)

    # add title, total and recognition to header
    header.insert(0,title)
    header.append("total")
    header.append("Recognition (%)")

    print(tabulate(matrix, headers=header))

def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    # creating labels if None
    if labels is None:
        labels = []
        for y in y_true:
            if y not in labels:
                labels.append(y)
    # creating pos_label if none
    if pos_label is None:
        pos_label = labels[0]

    tp = 0
    fp = 0
    for i in range(len(y_pred)):
        # if the prediction is positive
        if y_pred[i] == pos_label:
            # check if true positive
            if y_pred[i] == y_true[i]:
                tp += 1
            else:
                fp += 1
    # checks for zero case
    if tp + fp == 0:
        return 0
    precision = tp / (tp + fp)
    return precision

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    # creating labels if None
    if labels is None:
        labels = []
        for y in y_true:
            if y not in labels:
                labels.append(y)
    # creating pos_label if none
    if pos_label is None:
        pos_label = labels[0]

    tp = 0
    fn = 0
    for i in range(len(y_pred)):
        # if the prediction is positive
        if y_pred[i] == pos_label:
            # check if true positive
            if y_pred[i] == y_true[i]:
                tp += 1
        else:
            if y_true[i] == pos_label:
                fn += 1
    # checks for zero case
    if tp + fn == 0:
        return 0
    recall = tp / (tp + fn)
    return recall

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    numerator = 2* (binary_precision_score(y_true, y_pred, labels, pos_label) * binary_recall_score(y_true, y_pred, labels, pos_label))
    denominator = (binary_precision_score(y_true, y_pred, labels, pos_label) + binary_recall_score(y_true, y_pred, labels, pos_label))
    if denominator == 0:
        return 0
    return  numerator / denominator
