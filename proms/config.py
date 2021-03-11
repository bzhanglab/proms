import numpy as np
from scipy.sparse.construct import rand
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sklearn.linear_model import Ridge


"""
define package wise constants
"""

prediction_map = {
      'classification': 'cls',
      'regression': 'reg',
      'survival': 'sur'
  }


default_estimators = {
  'cls': ['lr', 'rf', 'svm', 'gbm'],
  'reg': ['ridge', 'rf', 'svm', 'gbm'],
#   'sur': ['coxph', 'rf', 'svm', 'gbm']
  'sur': ['coxph']
}

parameter_grid = {
    'cls': {
        'lr': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100]
        },
        'rf':{
            'max_depth': [2, 4, 6, 8, 10],
            'n_estimators': range(50, 400, 50)
        },
        'svm':{
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
        },
        'gbm':{
            'max_depth': [2, 4, 6, 8, 10],
            'n_estimators': range(50, 400, 50)
        }
    },
    'reg': {
        'ridge':{
            'alpha': 10. ** np.linspace(-4, 4, 10)
        },
        'rf':{
            'max_depth': [2, 4, 6, 8, 10],
            'n_estimators': range(50, 400, 50)
        },
        'svm':{
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
        },
        'gbm':{
            'max_depth': [2, 4, 6, 8, 10],
            'n_estimators': range(50, 400, 50)
        }
    },
    'sur': {
        'svm': {
            'alpha': 10. ** np.linspace(-4, 4, 10)
        },
        'coxph': {
            'alpha': 10. ** np.linspace(-4, 4, 10)
        },
        'gbm': {
            'max_depth': [2, 4, 6, 8, 10],
            'n_estimators': range(50, 400, 50)
        },
        'rf': {
            'max_depth': [2, 4, 6, 8, 10],
            'n_estimators': range(50, 400, 50)
        }
    }
}


def get_estimator(seed, estimator, prediction_type='cls'):
    """
    classfication:
        lr: sklearn.linear_model.LogisticRegression
        rf: sklearn.ensemble.RandomForestClassifier
        svm: sklearn.svm.SVC
        gbm: xgboost.XGBClassifier
    regression:
        rf: sklearn.ensemble.RandomForestRegressor
        svm: sklearn.svm.SVR
        gbm xgboost.XGBRegressor
    survival:
        coxph: sksurv.linear_model.CoxnetSurvivalAnalysis
        rf: sksurv.ensemble.RandomSurvivalForest
        svm: sksurv.svm.FastKernelSurvivalSVM
        gbm: sksurv.ensemble.GradientBoostingSurvivalAnalysis
    """
    if not estimator in default_estimators[prediction_type]:
        raise ValueError(f'estimator {estimator} not supported for prediction type {prediction_type}')

    if prediction_type == 'cls':
        est_dict = {
            'lr': LogisticRegression(max_iter=1000, random_state=seed),
            'rf': RandomForestClassifier(random_state=seed),
            'svm': SVC(kernel='rbf', probability=True, random_state=seed),
            'gbm': XGBClassifier(random_state=seed)
        }
        return est_dict[estimator]

    if prediction_type == 'reg':
        est_dict = {
            'ridge': Ridge(random_state=seed),
            'rf': RandomForestRegressor(random_state=seed),
            'svm': SVR(kernel='rbf'),
            'gbm': XGBRegressor(random_state=seed)
        }
        return est_dict[estimator]

    if prediction_type == 'sur':
        est_dict = {
            'coxph': CoxPHSurvivalAnalysis(),
            # 'rf': RandomSurvivalForest(random_state=seed),
            # 'svm': FastSurvivalSVM(random_state=seed),
            # 'gbm': GradientBoostingSurvivalAnalysis(random_state=seed)
        }
        return est_dict[estimator]
    
