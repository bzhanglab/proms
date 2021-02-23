import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression


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
  'reg': ['rf', 'svm', 'gbm'],
  'sur': ['coxph', 'rf', 'svm', 'gbm']
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
        'coxph': {
            'l1_ratio': np.arange(0, 1.1, 0.1).tolist()
        },
        'rf':{
            'max_depth': [2, 4, 6, 8, 10],
            'n_estimators': range(50, 400, 50)
        },
        'svm':{
            'kernel': ['linear', 'poly', 'rbf'],
            'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
        },
        'gbm':{
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

    if prediction_type == 'cls':
        if not estimator in default_estimators[prediction_type]:
            raise ValueError(f'estimator {estimator} not supported for '
                         'prediction type {prediction_type}')
        est_dict = {
            'lr': LogisticRegression(max_iter=1000, random_state=seed),
            'rf': RandomForestClassifier(random_state=seed),
            'svm': SVC(kernel='rbf', probability=True, random_state=seed),
            'gbm': XGBClassifier(random_state=seed)
        }
        return est_dict[estimator]

    if prediction_type == 'reg':
        if not estimator in default_estimators[prediction_type]:
            raise ValueError(f'estimator {estimator} not supported for '
                         'prediction type {prediction_type}')
        est_dict = {
            'rf': RandomForestRegressor(random_state=seed),
            'svm': SVR(kernel='rbf'),
            'gbm': XGBRegressor(random_state=seed)
        }
        return est_dict[estimator]

    # FIXME:
    # surv
    
