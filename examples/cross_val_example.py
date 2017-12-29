import turicreate as tc
from turicreate_cross_validation.cross_validation import shuffle_sframe, StratifiedKFold, cross_val_score

if __name__ == "__main__":
    url = 'https://static.turi.com/datasets/xgboost/mushroom.csv'
    sf = tc.SFrame.read_csv(url)
    sf['label'] = (sf['label'] == 'p')
    params = {'target': 'label'}
    sf = shuffle_sframe(sf)
    folds = StratifiedKFold(sf, 'label', 5)
    cross_val_score(folds, tc.random_forest_classifier.create, params)
