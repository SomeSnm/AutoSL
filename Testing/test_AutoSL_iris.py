import pytest
import pandas as pd
import glob
import os
from sklearn import datasets
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
import sys
sys.path.append('../')
from AutoSL import AutoSL


@pytest.fixture
def iris_setup():
    iris = datasets.load_iris()
    clf = ['AdaBoostClassifier', 'RandomForestClassifier', 'SVC',
           'DecisionTreeClassifier', 'KNeighborsClassifier', 'MultinomialNB']
    dim_r = ['SelectKBest', 'PCA', None]

    clf_p = [{'n_estimators': [10, 100], 'learning_rate': [0.001, 10]},
             {'n_estimators': [10, 100], 'max_depth': [2, 3]},
             {'kernel': ['rbf', 'poly', 'linear'], 'C': [0.5, 5, 10]},
             {'criterion': ['gini', 'entropy'], 'max_depth': [2, 3, 4]},
             {'n_neighbors': [3, 20], 'weights': ['uniform', 'distance']},
             {'alpha': [0.1, 10]}
             ]
    dim_r_p = [{'k': [2, 3], 'score_func': (chi2, f_classif)}, {'n_components': [2, 0.8]}, None]
    train_set, test_set, train_target, test_target = train_test_split(iris.data, iris.target, test_size=0.33,
                                                                      stratify=iris.target)
    tst = AutoSL(estimators=clf, estimator_params=clf_p,
                 dim_red=dim_r, dim_red_params=dim_r_p, scoring=['accuracy', 'f1_macro'])
    rst = tst.fit_predict(train_set, test_set, train_target, test_target, n_jobs=2)
    return rst


def test_numrows(iris_st):
    newest = max(glob.iglob('AutoSL_logs\\**'), key=os.path.getctime)
    with open(newest, 'r') as myfile:
        data = myfile.read().replace('\n', '')
    expected_lines = data.count(' - INFO - Estimator: ') - data.count('- ERROR - Failed to fit the model')
    assert iris_st.shape[0] == expected_lines


def test_accuracy_column(iris_setup):
    assert pd.isnull(iris_setup['accuracy']).any() is False


def test_min_accuracy(iris_setup):
    assert iris_setup['accuracy'].max() > 0.9

if __name__ == "__main__":
    iris = iris_setup()
    test_numrows(wd)
    test_accuracy_column(wd)
    test_min_accuracy(wd)