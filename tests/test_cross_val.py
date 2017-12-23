from anomalous_vertices_detection.utils.turi_utils import *
import turicreate as tc


def test_kfod_split_number():
    data = tc.SFrame({"id": range(100)})
    assert len(list(split_kfold(data, 10))) == 10
    assert len(list(split_kfold(data, 5))) == 5


def test_stratified_kfold_split_number():
    data = tc.SFrame({"id": range(100), 'label': [0] * 50 + [1] * 50})
    assert len(list(split_stratified_kfold(data, 'label', 10))) == 10
    assert len(list(split_stratified_kfold(data, 'label', 5))) == 5


def test_kfod_split_size():
    data = tc.SFrame({"id": range(100)})
    for train, test in split_kfold(data, 10):
        assert len(train) == 90
        assert len(test) == 10
    for train, test in split_kfold(data, 5):
        assert len(train) == 80
        assert len(test) == 20


def test_stratified_kfold_split_size():
    data = tc.SFrame({"id": range(100), 'label': [0] * 50 + [1] * 50})
    for train, test in split_stratified_kfold(data, 'label', 10):
        assert len(train) == 90
        assert len(test) == 10
    for train, test in split_stratified_kfold(data, 'label', 5):
        assert len(train) == 80
        assert len(test) == 20


def test_stratified_kfold_label_dist():
    data = tc.SFrame({"id": range(100), 'label': [0] * 50 + [1] * 50})
    for train, test in split_stratified_kfold(data, 'label', 10):
        assert len(train[train["label"] == 0]) == 45
        assert len(train[train["label"] == 1]) == 45
        assert len(test[test["label"] == 0]) == 5
        assert len(test[test["label"] == 1]) == 5
    for train, test in split_stratified_kfold(data, 'label', 5):
        assert len(train[train["label"] == 0]) == 40
        assert len(train[train["label"] == 1]) == 40
        assert len(test[test["label"] == 0]) == 10
        assert len(test[test["label"] == 1]) == 10
    data = tc.SFrame({"id": range(100), 'label': [0] * 90 + [1] * 10})
    for train, test in split_stratified_kfold(data, 'label', 10):
        assert len(train[train["label"] == 0]) == 81
        assert len(train[train["label"] == 1]) == 9
        assert len(test[test["label"] == 0]) == 9
        assert len(test[test["label"] == 1]) == 1


def test_kfold_split_unique():
    data = tc.SFrame({"id": range(100)})
    for train, test in split_kfold(data, 10):
        assert len(train) == len(train.unique())
        assert len(test) == len(test.unique())


def test_stratified_kfold_split_unique():
    data = tc.SFrame({"id": range(100), 'label': [0] * 50 + [1] * 50})
    for train, test in split_stratified_kfold(data, 'label', 10):
        assert len(train) == len(train.unique())
        assert len(test) == len(test.unique())


def test_kfold_split_intersect():
    data = tc.SFrame({"id": range(100)})
    for train, test in split_kfold(data, 10):
        assert 100 == len(train.unique().append(test.unique()))


def test_stratified_kfold_split_intersect():
    data = tc.SFrame({"id": range(100), 'label': [0] * 50 + [1] * 50})
    for train, test in split_stratified_kfold(data, 'label', 10):
        assert 100 == len(train.unique().append(test.unique()))