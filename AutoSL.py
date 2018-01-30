import pandas as pd
from sqlalchemy import create_engine
import collections
import time
import csv
import logging
import os
# Estimators
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.externals import joblib

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold


class AutoSL:
    """Initialize the class with the names of all Classifiers and """

    def __init__(self, estimators, estimator_params, dim_red=None, dim_red_params=None,
                 n_folds=None, scoring=None):
        """

        :param estimators: list
        List of estimator names
        :param estimator_params:  list of dictionaries
        List of dictionaries that contains parameters of passed estimators. Dictionaries should be in the same order as
        respective estimators. If None, the default parameters of estimators will be used.
        :param dim_red: list, default=None
        List of dimensionality reduction step names
        :param dim_red_params: list of dictionaries, default=None
        List of dictionaries that contains parameters of passed dimensionality reduction algorithms.
        :param n_folds: int or None, default=None
        Number of folds for cross validation. All folds are stratified by default. If None 3-fold will be used.
        :param scoring: string, list or None, default=None
        Scoring method(s) for given estimators. If None, the default will be used.
        """
        self._clf_mapping = {'AdaBoostClassifier': AdaBoostClassifier(),
                             'RandomForestClassifier': RandomForestClassifier(),
                             'SVC': SVC(),
                             'DecisionTreeClassifier': DecisionTreeClassifier(),
                             'MultinomialNB': MultinomialNB(),
                             'KNeighborsClassifier': KNeighborsClassifier()
                             }

        self._dim_red_mapping = {'SelectKBest': SelectKBest(),
                                 'PCA': PCA(),
                                 None: None
                                 }
        self._estimators = []
        for est in estimators:
            if est not in self._clf_mapping.keys():  # Making sure that all estimators are spelled correctly
                raise ValueError(str(est), ' is not among available estimators')
            else:
                self._estimators.append(self._clf_mapping[est])

        self._dim_red = []
        if dim_red is None:  # If there is no dimensionality reduction step, we create list with one element of None
            self._dim_red = [None]
            self._dim_red_params = [None]
        else:
            for prep in dim_red:
                if prep not in self._dim_red_mapping.keys():
                    raise ValueError(str(prep), ' is not among available estimators')
                else:
                    self._dim_red.append(self._dim_red_mapping[prep])
            self._dim_red_params = dim_red_params

        self._estimator_params = estimator_params

        if len(self._estimators) != len(self._estimator_params):
            raise ValueError("The length of estimator list should be equal to the length of estimator_params list")

        if len(self._dim_red) != len(self._dim_red_params):
            raise ValueError("The length of dim_red list should be equal to the length of dim_red_params list")

        if scoring is None:
            self._scoring = [None]
        elif isinstance(scoring, str):
            self._scoring = [scoring]
        elif isinstance(scoring, collections.Iterable):
            self._scoring = scoring
        else:
            raise ValueError("The scoring is not a string, None or iterable")

        self._n_folds = n_folds

        self._best_models = pd.DataFrame(columns=['scorer', 'score', 'model'])  # DataFrame that contains best models
        self._result_table = None  # Result table

        # Setting up logging:
        logging.basicConfig(level=logging.INFO)
        self._logger = logging.getLogger(__name__)
        if not os.path.exists('AutoSL_logs'):  # Creates the folder if it does not exist
            os.makedirs('AutoSL_logs')
        handler = logging.FileHandler('./AutoSL_logs/AutoSL_'+time.strftime("%Y-%m-%d--%H-%M-%S")+'.log')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)

    @staticmethod
    def _evaluate_model(model, test_set, y_test):
        """
        Produce the evaluation report of a model
        :param model: Pipeline
        Trained Pipeline model.
        :param test_set: array-like, shape = [n_samples, n_features]
        Test features
        :param y_test:array-like, labels of n_samples
        Test labels
        """
        num_classes = len(set(y_test))
        y_predict = model.predict(test_set)
        cv_res = pd.DataFrame(model.cv_results_)
        nb_candidates = len(cv_res.index)
        cv_res = cv_res.iloc[model.best_index_]

        try:  # Checking if pipeline contains dimensionality reduction step
            dr_name = model.best_estimator_.named_steps['dr'].__class__.__name__
            methods = {'scoring': model.scoring,
                       'cv_seeds': model.cv,
                       'candidates_number': nb_candidates,
                       'reduce_dim': dr_name,
                       'classifier': model.best_estimator_.named_steps['clf'].__class__.__name__}
        except:
            methods = {'scoring': model.scoring,
                       'cv_seeds': model.cv,
                       'candidates_number': nb_candidates,
                       'reduce_dim': None,
                       'classifier': model.best_estimator_.named_steps['clf'].__class__.__name__}

        parameters = {}
        for key in cv_res['params']:
            if callable(cv_res['params'][key]):
                parameters.update({key: cv_res['params'][key].__name__})
            else:
                parameters.update({key: cv_res['params'][key]})

        if num_classes > 2:  # If we have multiclass problem
            acc = accuracy_score(y_test, y_predict)
            f1_score_macro = f1_score(y_test, y_predict, average='macro')
            precision_macro = precision_score(y_test, y_predict, average='macro')
            recall_macro = recall_score(y_test, y_predict, average='macro')

            scores = {'accuracy': acc,
                      'f1_score[macro]': f1_score_macro,
                      'precision[macro]': precision_macro,
                      'recall[macro]': recall_macro
                      }

        elif num_classes == 2:  # If we have binary classification problem
            acc = accuracy_score(y_test, y_predict)
            mcc = matthews_corrcoef(y_test, y_predict)
            f1_score_macro = f1_score(y_test, y_predict, average='macro')
            precision_macro = precision_score(y_test, y_predict, average='macro')
            recall_macro = recall_score(y_test, y_predict, average='macro')
            confm = confusion_matrix(y_test, y_predict, labels=[0, 1])
            tpr = confm[0][0] / float(confm[0][0] + confm[0][1])
            tnr = confm[1][1] / float(confm[1][1] + confm[1][0])
            roc_auc = roc_auc_score(y_test, y_predict)
            scores = {'accuracy': acc,
                      'mcc': mcc,
                      'f1_score[macro]': f1_score_macro,
                      'precision[macro]': precision_macro,
                      'recall[macro]': recall_macro,
                      'true_positive_rate': tpr,
                      'true_negative_rate': tnr,
                      'confm_tp': confm[0][0],
                      'confm_fp': confm[0][1],
                      'confm_fn': confm[1][0],
                      'confm_tn': confm[1][1],
                      'roc_auc': roc_auc}
        else:
            raise ValueError("The classification should contain at least 2 target classes")

        train_info = {'mean_fit_time': cv_res['mean_fit_time'],
                      'mean_score_time': cv_res['mean_score_time'],
                      'mean_test_score': cv_res['mean_test_score'],
                      'mean_train_score': cv_res['mean_train_score']}

        info = {}
        info.update(methods)
        info.update(parameters)
        info.update(scores)
        info.update(train_info)
        results = pd.DataFrame(info, index=[0])

        return results

    def _create_params(self, est_n, dimr_n):
        """
        Create dictionary of parameters for given pipeline with proper naming.

        :param est_n: int
        Index of a current estimator in provided estimator list.
        :param dimr_n: int
        Index of a current dimensionality reduction algorithm in provided list.
        """
        params = {}
        for key, value in self._estimator_params[est_n].items():
            params['clf__' + key] = value

        if self._dim_red_params[dimr_n] is None:
            return params
        else:
            for key, value in self._dim_red_params[dimr_n].items():
                params['dr__' + key] = value
            return params

    def fit_predict(self, x_train, x_test, y_train, y_test, n_jobs=-1):

        """
        Training all dimensionality reduction and estimator combinations and recoding their performance metrics.

        :param n_jobs: int, default = -1
        Number of jobs to run in parallel. Default "-1" uses all available computational power.
        :return:
        :rtype: DataFrame
        :param x_train: array-like, shape = [n_samples, n_features]
        :param x_test: array-like, shape = [n_samples]
        :param y_train: array-like, shape = [n_samples, n_features]
        :param y_test: array-like, shape = [n_samples]
        """

        start_time = time.time()
        result_table = pd.DataFrame()
        for sc in self._scoring:
            for dr in range(len(self._dim_red)):
                for est in range(len(self._estimators)):
                    if self._dim_red[dr] is None:  # Checking if there is a dimensionality reduction step
                        ppl = Pipeline([('clf', self._estimators[est])])
                        self._logger.info("Estimator: " + self._estimators[est].__class__.__name__ +
                                          ", Dimensionality Reduction: " + 'None' + ", Scorer: " + str(sc))
                    else:
                        ppl = Pipeline([('dr', self._dim_red[dr]), ('clf', self._estimators[est])])
                        self._logger.info("Estimator: " + self._estimators[est].__class__.__name__ +
                                          ", Dimensionality Reduction: " + self._dim_red[dr].__class__.__name__ +
                                          ", Scorer: " + str(sc))

                    params = self._create_params(est_n=est, dimr_n=dr)

                    grd = GridSearchCV(estimator=ppl, param_grid=params,
                                       scoring=sc, cv=self._n_folds, verbose=1, n_jobs=n_jobs)
                    try:  # Catching cases when the model can't be trained and writing error message to a log file
                        grd.fit(x_train, y_train)
                        res = self._evaluate_model(grd, x_test, y_test)
                        self._logger.info("Accuracy on test set: " + str(res.iloc[0, 0]))
                        result_table = result_table.append(res, ignore_index=True)
                        result_table.to_csv('AutoSL_results.csv', index=False)
                        best = [sc, grd.best_score_, grd.best_estimator_]
                        self._best_models = self._best_models.append(pd.DataFrame([best],
                                                                                  columns=['scorer', 'score', 'model'],
                                                                                  index=[0]),
                                                                     ignore_index=True)
                    except Exception as e:
                        self._logger.error('Failed to fit the model with following error: ' + str(e))
                    self._logger.info('#'*100)
            print("=" * 100)
            indm = self._best_models[self._best_models['scorer'] == sc]['score'].idxmax()  # Index of max value
            print("Scorer: ", sc, ", best score: ", self._best_models.iloc[indm, 1])
            print("Model with best score: ", self._best_models.iloc[indm, 2])
            print("=" * 100)
        fin_t = time.time() - start_time
        self._logger.info('Total time: ' + str(fin_t/60) + ' minutes')
        self._result_table = result_table
        return result_table

    def export_best_model(self, filename=None, scorer='accuracy'):
        """
        Saves the best performing model to a file according to a given scorer. The joblib package was used to save the
        scikit-learn model.

        :param filename: string
        File name to save.

        :param scorer: string
        The scorer name. Note: Only scoring methods used in fit_predict step should be passed.
        """
        scrs = set(self._best_models['scorer'].values)
        if scorer not in scrs:
            raise KeyError("The scorer ", scorer, " is not available. The following scorers are available:", list(scrs))
        scdf = self._best_models[self._best_models['scorer'] == scorer]
        mdl = scdf[scdf['score'] == scdf['score'].max()]['model'].iloc[0]
        print("Model: ", mdl)
        print("Score: ", scdf['score'].max())
        if filename is not None:
            joblib.dump(mdl, filename)
        return mdl

    def results_to_sql(self, username, password, host_address, db_name, schema, table_name, port=5432):

        """
        Saves the result table to a table in a database.

        Database connection credentials:
        :param username: string
        :param password: string
        :param host_address: string
        :param db_name: string
        :param schema: string
        :param table_name: string
        :param port: int
        """
        engine = create_engine('postgresql://' + str(username) + ':' + str(password) + '@'
                               + str(host_address) + ':' + str(port) + '/' + str(db_name))

        self._result_table.to_sql(table_name,  # Writing the result table to a database
                                  con=engine,
                                  schema=schema,
                                  if_exists='replace',
                                  index=False,
                                  index_label=None,
                                  chunksize=None,
                                  dtype=None)
