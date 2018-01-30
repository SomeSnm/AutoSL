import pytest
import pandas as pd
import glob
import os
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
import sys
sys.path.append('../')
from AutoSL import AutoSL


@pytest.fixture
def wine_setup():
    cnames = ["Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids",
              "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines",
              "Proline"]
    wine = pd.read_csv('wine.data', header=None, names=cnames, index_col=False)
    wine = wine[wine['Alcohol'].isin([1, 2])]  # Making the problem binary
    clf = ['AdaBoostClassifier', 'RandomForestClassifier', 'SVC',
           'DecisionTreeClassifier', 'KNeighborsClassifier', 'MultinomialNB']
    dim_r = ['SelectKBest', 'PCA', None]

    clf_p = [{'n_estimators': [10, 100], 'learning_rate': [0.001, 10]},
             {'n_estimators': [10, 100], 'max_depth': [3,  10]},
             {'kernel': ['rbf', 'poly', 'linear']},
             {'criterion': ['gini', 'entropy'], 'max_depth': [4, 12]},
             {'n_neighbors': [5, 50], 'weights': ['uniform', 'distance']},
             {'alpha': [0.1, 10]}
             ]
    dim_r_p = [{'k': [4, 12], 'score_func': (chi2, f_classif)}, {'n_components': [3, 0.8]}, None]
    train_set, test_set, train_target, test_target = train_test_split(wine.iloc[:, 1:], wine.iloc[:, 0], test_size=0.33,
                                                                      stratify=wine.iloc[:, 0])
    tst = AutoSL(estimators=clf, estimator_params=clf_p, dim_red=dim_r, dim_red_params=dim_r_p,
                 scoring=['accuracy', 'f1_macro'], n_folds=4)
    rst = tst.fit_predict(train_set, test_set, train_target, test_target)

    return rst


def test_numrows(wine_st):
    newest = max(glob.iglob('AutoSL_logs\\**'), key=os.path.getctime)
    with open(newest, 'r') as myfile:
        data = myfile.read().replace('\n', '')
    expected_lines = data.count(' - INFO - Estimator: ') - data.count('- ERROR - Failed to fit the model')
    assert wine_st.shape[0] == expected_lines
    print('Number of rows tested successfully')


def test_accuracy_column(wine_st):
    assert pd.isnull(wine_st['accuracy']).any() == False


def test_min_accuracy(wine_st):
    assert wine_st['accuracy'].max() > 0.9


if __name__ == "__main__":
    wd = wine_setup()
    test_numrows(wd)
    test_accuracy_column(wd)
    test_min_accuracy(wd)
