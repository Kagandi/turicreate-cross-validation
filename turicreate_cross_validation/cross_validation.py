# -*- coding: utf-8 -*-
"""
This module API is mostly compatible with the old graphlab-create cross_validation module.
https://turi.com/products/create/docs/graphlab.toolkits.cross_validation.html
"""
import numpy as np
import turicreate as tc
from collections import defaultdict
from turicreate.toolkits._internal_utils import _raise_error_if_column_exists
from turicreate.toolkits._main import ToolkitError


def _get_classification_metrics(model, targets, predictions):
    """

    Parameters
    ----------
    model: Classifier
        Turi create trained classifier.
    targets: SArray
        Array containing the expected labels.
    predictions: SArray
        Array containing the predicted labels.

    Returns
    -------
    dict
        An average metrics of the n folds.

    """
    precision = tc.evaluation.precision(targets, predictions)
    accuracy = tc.evaluation.accuracy(targets, predictions)
    recall = tc.evaluation.recall(targets, predictions)
    auc = tc.evaluation.auc(targets, predictions)
    return {"recall": recall,
            "precision": precision,
            "accuracy": accuracy,
            "auc": auc
            }


def _kfold_sections(data, n_folds):
    """
    Calculate the indexes of the splits that should
    be used to split the data into n_folds.

    Parameters
    ----------
    data: SFrame
        A Non empty SFrame.
    n_folds: int
        The number of folds to create. Must be at least 2.


    Yields
    -------
    (int, int)
        Yields the first and last index of the fold.

    Notes
    -----
        Based on scikit implementation.
    """
    Neach_section, extras = divmod(len(data), n_folds)
    section_sizes = ([0] +
                     extras * [Neach_section + 1] +
                     (n_folds - extras) * [Neach_section])
    div_points = np.array(section_sizes).cumsum()
    for i in range(n_folds):
        st = div_points[i]
        end = div_points[i + 1]
        yield st, end


def shuffle_sframe(sf, random_seed=None, temp_shuffle_col="shuffle_col"):
    """
    Create a copy of the SFrame where the rows have been shuffled randomly.

    Parameters
    ----------
    sf: SFrame
        A Non empty SFrame.
    random_seed: int, optional
        Random seed to use for the randomization. If provided, each call
        to this method will produce an identical result.
    temp_shuffle_col: str, optional
        Change only if you use the same column name.

    Returns
    -------
    SFrame
        A randomly shuffled SFrame.

    Examples
    --------
        >>> url = 'https://static.turi.com/datasets/xgboost/mushroom.csv'
        >>> sf = tc.SFrame.read_csv(url)
        >>> shuffle_sframe(sf)
    """

    if temp_shuffle_col in sf.column_names():
        raise ToolkitError('The SFrame already contains column named {0}. '
                           'Please enter set another value to temp_shuffle_col'.format(temp_shuffle_col))
    shuffled_sframe = sf.copy()
    shuffled_sframe[temp_shuffle_col] = tc.SArray.random_integers(sf.num_rows(), random_seed)
    return shuffled_sframe.sort(temp_shuffle_col).remove_column(temp_shuffle_col)


def KFold(data, n_folds=10):
    """
    Create a K-Fold split of a data set as an iterable/indexable object of K pairs,
    where each pair is a partition of the dataset.  This can be useful for cross
    validation, where each fold is used as a held out dataset while training
    on the remaining data.

    Parameters
    ----------
    data: SFrame
        A Non empty SFrame.
    n_folds: int
        The number of folds to create. Must be at least 2.

    Notes
    -----
    This does not shuffle the data. Shuffling your data is a useful preprocessing step when doing cross validation.

    Yields
    -------
    (SArray, SArray)
        Yields train, test of each fold

    Examples
    --------
        >>> url = 'https://static.turi.com/datasets/xgboost/mushroom.csv'
        >>> sf = tc.SFrame.read_csv(url)
        >>> folds = KFold(sf)
    """
    if data.num_rows() < n_folds:
        raise ValueError
    for st, end in _kfold_sections(data, n_folds):
        idx = np.zeros(len(data))
        idx[st:end] = 1
        yield data[tc.SArray(1 - idx)], data[tc.SArray(idx)]


def StratifiedKFold(data, label='label', n_folds=10):
    """
    Create a Starified K-Fold split of a data set as an iteratable/indexable object
    of K pairs, where each pair is a partition of the data set. This can be useful
    for cross validation, where each fold is used as a heldout dataset while
    training on the remaining data. Unlike the regular KFold the folds are
    made by preserving the percentage of samples for each class.
    The StratifiedKFold is more suitable for smaller datasets
    or for datasets where there a is a minority class.

    Parameters
    ----------
    data: SFrame
        A Non empty SFrame.
    label: str
        The target/class column name in the SFrame.
    n_folds: int
        The number of folds to create. Must be at least 2.

    Notes
    -----
    This does not shuffle the data. Shuffling your data is a useful preprocessing step when doing cross validation.

    Yields
    -------
    (SArray, SArray)
        Yields train, test of each fold

    Examples
    --------
        >>> url = 'https://static.turi.com/datasets/xgboost/mushroom.csv'
        >>> sf = tc.SFrame.read_csv(url)
        >>> folds = StratifiedKFold(sf)
    """
    _raise_error_if_column_exists(data, label, 'data', label)

    labels = data[label].unique()
    labeled_data = [data[data[label] == l] for l in labels]
    fold = [KFold(item, n_folds) for item in labeled_data]
    for _ in range(n_folds):
        train, test = tc.SFrame(), tc.SFrame()
        for f in fold:
            x_train, x_test = f.next()
            train = train.append(x_train)
            test = test.append(x_test)
        yield train, test


def cross_val_score(datasets, model_factory, model_parameters=None, evaluator=_get_classification_metrics):
    """
    Evaluate model performance via cross validation for a given set of parameters.

    Parameters
    ----------
    datasets: iterable of tuples
        The data used to train the model on a format of iterable of tuples.
        The tuples should be in a format of (train, test).
    model_factory: function
        This is the function used to create the model.
        For example, to perform model_parameter_search using the
        GraphLab Create model graphlab.linear_regression.LinearRegression,
        the model_factory is graphlab.linear_regression.create().
    model_parameters: dict
        The params argument takes a dictionary containing parameters that will be passed to the provided model factory.
    evaluator: function (model, training_set, validation_set) -> dict, optional
        The evaluation function takes as input the model, training and validation SFrames,
        and returns a dictionary of evaluation metrics where each value is a simple type, e.g. float, str, or int.

    Returns
    -------
    dict
        The calculate metrics for the cross validation.

    Examples
    --------
        >>> url = 'https://static.turi.com/datasets/xgboost/mushroom.csv'
        >>> sf = tc.SFrame.read_csv(url)
        >>> folds = StratifiedKFold(sf)
        >>> params = {'target': 'label'}
        >>> cross_val_score(folds, tc.random_forest_classifier.create, params)
    """
    if not model_parameters:
        model_parameters = {'target': 'label'}
    label = model_parameters['target']

    cross_val_metrics = defaultdict(list)
    for train, test in datasets:

        _raise_error_if_column_exists(train, label, 'train', label)
        _raise_error_if_column_exists(test, label, 'test', label)

        model = model_factory(train, **model_parameters)
        prediction = model.predict(test)
        metrics = evaluator(model, test[label], prediction)
        for k, v in metrics.iteritems():
            cross_val_metrics[k].append(v)
    return {k: np.mean(v) for k, v in cross_val_metrics.iteritems()}
