# turicreate-cross-validation
Implementation  of cross validation for turicreate.
This module was developed to supply a cross-validation functions to turicreate, until an official version will be implemnted.  
The API and stingdocs are mostly based on the [old version of turicrate](https://turi.com/products/create/docs/graphlab.toolkits.cross_validation.html).

## Usage
```python
from turicreate_cross_validation.cross_validation import *

url = 'https://static.turi.com/datasets/xgboost/mushroom.csv'
sf = tc.SFrame.read_csv(url)
sf['label'] = (sf['label'] == 'p')
params = {'target': 'label'}
sf = shuffle_sframe(sf)
folds = StratifiedKFold(sf, 'label', 5)
cross_val_score(folds, tc.random_forest_classifier.create, params, label='label')
```

## Credits
* [Guy Rapport](https://github.com/guy4261) for the feedback and advices to make this code better.
