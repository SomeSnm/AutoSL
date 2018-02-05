# AutoSL
The script automatically searches for best performing classifier wih dimensionality reduction techniques from a given set. The results are saved in a final result table and can be uploaded to the sql database. The script also produces the logs that are saved in a separate folder. 

Example use:
```python
from sklearn.model_selection import train_test_split
from sklearn import datasets
from AutoSl import AutoSL


X = datasets.load_breast_cancer()['data']
y = datasets.load_breast_cancer()['target']

train_set, test_set, train_target, test_target = train_test_split(X, y, test_size=0.33, stratify = y)

tst = AutoSL(estimators=['RandomForestClassifier', 'AdaBoostClassifier'], 
             estimator_params = [{'n_estimators': [5, 10], 'max_depth': [5, 30]}, 
                                 {'n_estimators': [10, 100], 'learning_rate': [0.01, 10]}],
             scoring = ['accuracy', 'f1_macro','roc_auc'],
             dim_red = ['PCA',None], dim_red_params = [{'n_components':[10, 0.8]}, None], name='cancer_data')
result = tst.fit_predict(train_set, test_set, train_target, test_target)
```
