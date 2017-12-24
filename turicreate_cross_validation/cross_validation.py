"""
This module API is compatible with the old graphlab-create cross_validation module.
https://turi.com/products/create/docs/graphlab.toolkits.cross_validation.html
"""
import numpy as np
import turicreate as tc
from collections import defaultdict


def shuffle_sframe(sf, random_seed=None):
    sf["shuffle_col"] = tc.SArray.random_integers(sf.num_rows(), random_seed)
    return sf.sort("shuffle_col").remove_column("shuffle_col")


def kfold_sections(data, n_folds):
    """
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


def split_kfold(data, n_folds=10):
    for st, end in kfold_sections(data, n_folds):
        idx = np.zeros(len(data))
        idx[st:end] = 1
        yield data[tc.SArray(1 - idx)], data[tc.SArray(idx)]


def split_stratified_kfold(data, label='label', n_folds=10):
    if label in data.column_names():
        labels = data[label].unique()
        labeled_data = [data[data[label] == l] for l in labels]
        fold = [split_kfold(item, n_folds) for item in labeled_data]
        for _ in range(n_folds):
            train, test = tc.SFrame(), tc.SFrame()
            for f in fold:
                x_train, x_test = f.next()
                train = train.append(x_train)
                test = test.append(x_test)
            yield train, test
    else:
        yield split_kfold(data, n_folds)


def cross_validate(datasets, model_factory, model_parameters=None, evaluator=get_classification_metrics, label='label'):
    if not model_parameters:
        model_parameters = {}
    cross_val_metrics = defaultdict(list)
    for train, test in datasets:
        model = model_factory(train, **model_parameters)
        prediction = model.predict(test)
        metrics = evaluator(model, test[label], prediction)
        for k, v in metrics.iteritems():
            cross_val_metrics[k].append(v)
    return {k: np.mean(v) for k, v in cross_val_metrics.iteritems()}