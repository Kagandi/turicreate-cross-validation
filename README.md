# turicreate-cross-validation
Implementation  of cross validation for turicreate.
This module was developed to supply a cross-validation functions to turicreate, until an official version will be implemnted.  
The API and docstrings are mostly based on the [old version of turicrate](https://turi.com/products/create/docs/graphlab.toolkits.cross_validation.html).

## Usage
Install using: ```pip install -e git+https://github.com/Kagandi/turicreate-cross-validation.git#egg=turicreate_cross_validation```
Since the project is pretty small you also can just use cross_validation.py in your project. 
```python
import turicreate as tc
from turicreate_cross_validation import shuffle_sframe, StratifiedKFold, cross_val_score

url = 'https://static.turi.com/datasets/xgboost/mushroom.csv'
sf = tc.SFrame.read_csv(url)
sf['label'] = (sf['label'] == 'p')
params = {'target': 'label', "verbose": False}
sf = shuffle_sframe(sf)
folds = StratifiedKFold(sf, 'label', 5)
print(cross_val_score(folds, tc.random_forest_classifier.create, params))
```

## Credits
* [Guy Rapport](https://github.com/guy4261) for the feedback and advices to make this code better.
