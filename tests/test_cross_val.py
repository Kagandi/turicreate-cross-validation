from itertools import izip
from turicreate_cross_validation.cross_validation import KFold, StratifiedKFold, cross_val_score, shuffle_sframe, \
    _kfold_sections
import turicreate as tc
import pytest
from turicreate.toolkits._main import ToolkitError


def test_kfold_split_number():
    data = tc.SFrame({"id": range(100)})
    assert len(list(KFold(data, 10))) == 10
    assert len(list(KFold(data, 5))) == 5


def test_stratified_kfold_split_number():
    data = tc.SFrame({"id": range(100), 'label': [0] * 50 + [1] * 50})
    assert len(list(StratifiedKFold(data, 'label', 10))) == 10
    assert len(list(StratifiedKFold(data, 'label', 5))) == 5


def test_kfold_split_size():
    data = tc.SFrame({"id": range(100)})
    for train, test in KFold(data, 10):
        assert len(train) == 90
        assert len(test) == 10
    for train, test in KFold(data, 5):
        assert len(train) == 80
        assert len(test) == 20


def test_stratified_kfold_split_size():
    data = tc.SFrame({"id": range(100), 'label': [0] * 50 + [1] * 50})
    for train, test in StratifiedKFold(data, 'label', 10):
        assert len(train) == 90
        assert len(test) == 10
    for train, test in StratifiedKFold(data, 'label', 5):
        assert len(train) == 80
        assert len(test) == 20


def test_stratified_kfold_label_dist():
    data = tc.SFrame({"id": range(100), 'label': [0] * 50 + [1] * 50})
    for train, test in StratifiedKFold(data, 'label', 10):
        assert len(train[train["label"] == 0]) == 45
        assert len(train[train["label"] == 1]) == 45
        assert len(test[test["label"] == 0]) == 5
        assert len(test[test["label"] == 1]) == 5
    for train, test in StratifiedKFold(data, 'label', 5):
        assert len(train[train["label"] == 0]) == 40
        assert len(train[train["label"] == 1]) == 40
        assert len(test[test["label"] == 0]) == 10
        assert len(test[test["label"] == 1]) == 10
    data = tc.SFrame({"id": range(100), 'label': [0] * 90 + [1] * 10})
    for train, test in StratifiedKFold(data, 'label', 10):
        assert len(train[train["label"] == 0]) == 81
        assert len(train[train["label"] == 1]) == 9
        assert len(test[test["label"] == 0]) == 9
        assert len(test[test["label"] == 1]) == 1


def test_kfold_split_unique():
    data = tc.SFrame({"id": range(100)})
    for train, test in KFold(data, 10):
        assert len(train) == len(train.unique())
        assert len(test) == len(test.unique())


def test_stratified_kfold_split_unique():
    data = tc.SFrame({"id": range(100), 'label': [0] * 50 + [1] * 50})
    for train, test in StratifiedKFold(data, 'label', 10):
        assert len(train) == len(train.unique())
        assert len(test) == len(test.unique())


def test_kfold_split_intersect():
    data = tc.SFrame({"id": range(100)})
    for train, test in KFold(data, 10):
        assert 100 == len(train.unique().append(test.unique()))


def test_stratified_kfold_split_intersect():
    data = tc.SFrame({"id": range(100), 'label': [0] * 50 + [1] * 50})
    for train, test in StratifiedKFold(data, 'label', 10):
        assert 100 == len(train.unique().append(test.unique()))


def test_cross_val_score_with_wrong_label():
    data = tc.SFrame({"id": range(100), 'label': [0] * 50 + [1] * 50})
    params = {'target': 'label2'}
    folds = StratifiedKFold(data, 'label', 5)
    with pytest.raises(ToolkitError):
        cross_val_score(folds, tc.random_forest_classifier.create, params)


def test_StratifiedKFold_with_wrong_label():
    data = tc.SFrame({"id": range(100), 'label': [0] * 50 + [1] * 50})
    with pytest.raises(ToolkitError):
        folds = StratifiedKFold(data, 'label2', 5)
        for train, test in folds:
            pass


def test_cross_val_basic():
    data = tc.SFrame({"id": ["a"] * 50 + ["b"] * 50, 'label': [0] * 50 + [1] * 50})
    params = {'target': 'label'}
    folds = StratifiedKFold(data, 'label', 5)
    metrics = cross_val_score(folds, tc.decision_tree_classifier.create, params)
    assert metrics == {'recall': 1.0, 'auc': 1.0, 'precision': 1.0, 'accuracy': 1.0}


def test_shuffle_sframe_id_different():
    data = tc.SFrame({"id": range(100), 'id2': range(100)})
    equal_counter = 0
    shuffled_sframe = shuffle_sframe(data)
    assert len(shuffled_sframe) == len(data)
    for item, shuffeled_item in izip(data, shuffled_sframe):
        if item == shuffeled_item:
            equal_counter += 1
    assert equal_counter != len(data)


def test_shuffle_sframe_same_items():
    data = tc.SFrame({"id": range(100), 'id2': range(100)})
    shuffled_sframe = shuffle_sframe(data)
    shuffled_sframe = shuffled_sframe.sort("id")
    for item, shuffeled_item in izip(data, shuffled_sframe):
        assert item == shuffeled_item


def test_kfold_sections():
    data = tc.SFrame({"id": range(100), 'id2': range(100)})
    prev_end = None
    for st, end in _kfold_sections(data, 10):
        assert end - st == 10
        if prev_end:
            assert prev_end == st
        prev_end = end
