import multiprocessing
import warnings
import os
import json
import yaml
import csv
from datetime import datetime
from tempfile import mkdtemp
from shutil import rmtree
import pandas as pd
import numpy as np
import argparse
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
# from .utils import StandardScalerDf
from joblib import Memory
from proms import Dataset, FeatureSelector, config


def check_prediction_type_ok(prediction_type):
    return True if prediction_type in ['cls', 'reg', 'sur'] else False


def get_scores(prediction_type, grid, X_test, y_test, X_test_2, y_test_2):
    """
     evaluate on the test data set (if available)
     and independent test set (if available)
    """
    scores = None
    if prediction_type == 'cls':
        test_acc = test_score = 'NA'
        label = pred_label_s = pred_prob_s = 'NA'
        test_acc_2 = test_score_2 = 'NA'
        label_2 = pred_label_s_2 = pred_prob_s_2 = 'NA'
        if X_test is not None:
            label = ','.join(map(str, y_test))
            pred_prob = grid.predict_proba(X_test)[:, 1]
            pred_label = [1 if x >= 0.5 else 0 for x in pred_prob]
            test_score = grid.score(X_test, y_test)
            pred_prob_s = ','.join(map('{:.4f}'.format, pred_prob))
            pred_label_s = ','.join(map(str, pred_label))
            test_acc = accuracy_score(y_test, pred_label)

        if X_test_2 is not None:
            pred_prob_2 = grid.predict_proba(X_test_2)[:, 1]
            pred_label_2 = [1 if x >= 0.5 else 0 for x in pred_prob_2]
            pred_prob_s_2 = ','.join(map('{:.4f}'.format, pred_prob_2))
            pred_label_s_2 = ','.join(map(str, pred_label_2))
            if y_test_2 is not None:
                label_2 = ','.join(map(str, y_test_2))
                test_score_2 = round(grid.score(
                    X_test_2, y_test_2), 4)
                test_acc_2 = round(accuracy_score(y_test_2, pred_label_2), 4)
        scores = {'test_acc': test_acc,
                  'test_score': test_score,
                  'pred_prob_s': pred_prob_s,
                  'pred_label_s': pred_label_s,
                  'label': label,
                  'pred_prob_s_2': pred_prob_s_2,
                  'pred_label_s_2': pred_label_s_2,
                  'label_2': label_2,
                  'test_acc_2': test_acc_2,
                  'test_score_2': test_score_2
                 }
    elif prediction_type == 'reg':
        test_mse = test_score = 'NA'
        label = pred_label_s = 'NA'
        test_mse_2 = test_score_2 = 'NA'
        label_2 = pred_label_s_2 = 'NA'
        if X_test is not None:
            label = ','.join(map(str, y_test))
            pred_label = grid.predict(X_test)
            pred_label = [round(item, 4) for item in pred_label]
            test_score = grid.score(X_test, y_test)
            pred_label_s = ','.join(map(str, pred_label))
            test_mse = mean_squared_error(y_test, pred_label)

        if X_test_2 is not None:
            pred_label_2 = grid.predict(X_test_2)
            pred_label_s_2 = ','.join(map(str, pred_label_2))
            if y_test_2 is not None:
                label_2 = ','.join(map(str, y_test_2))
                test_score_2 = round(grid.score(X_test_2, y_test_2), 4)
                test_mse_2 = round(mean_squared_error(y_test_2, pred_label_2), 4)
        scores = {'test_mse': test_mse,
                  'test_score': test_score,
                  'pred_label_s': pred_label_s,
                  'label': label,
                  'pred_label_s_2': pred_label_s_2,
                  'label_2': label_2,
                  'test_mse_2': test_mse_2,
                  'test_score_2': test_score_2
                 }
    elif prediction_type == 'sur':
        test_risk_score_s = test_risk_score_s_2 = 'NA'
        test_c_index = test_c_index_2 = 'NA'
        label = label_2 = 'NA'
        if X_test is not None:
            label = ','.join(map(str, y_test))
            test_risk_score = grid.predict(X_test)
            test_c_index = round(grid.score(X_test, y_test), 4)
            test_risk_score_s = ','.join(map('{:.4f}'.format, test_risk_score))

        if X_test_2 is not None:
            test_risk_score_2 = grid.predict(X_test_2)
            test_risk_score_s_2 = ','.join(map('{:.4f}'.format, test_risk_score_2))
            if y_test_2 is not None:
                label_2 = ','.join(map(str, y_test_2))
                test_c_index_2 = round(grid.score(X_test_2, y_test_2), 4)
        scores = {'test_risk_score': test_risk_score_s,
                  'label': label,
                  'test_score': test_c_index,
                  'test_risk_score_2': test_risk_score_s_2,
                  'label_2': label_2,
                  'test_score_2': test_c_index_2
                 }
    
    return scores


def set_up_results(data, mode, run_config, prediction_type, fs_method,
                   k, estimator, repeat, scores, grid):
    best_fs = grid.best_estimator_.named_steps['featureselector']
    selected_features = best_fs.get_feature_names()
    res = None

    if mode == 'full':
        cluster_membership = best_fs.get_cluster_membership()
        run_version = run_config['run_version']
        output_root = run_config['output_root']
        out_dir_run = os.path.join(output_root, run_version)
        out_dir_run_full = os.path.join(out_dir_run, 'full_model')
        with open(os.path.join(out_dir_run_full, 'full_model.pkl'), 'wb') as fh:
            pickle.dump(grid.best_estimator_, fh,
                        protocol=pickle.HIGHEST_PROTOCOL)

    omics_type = 'so' if fs_method == 'proms' else 'mo'
    if mode == 'eval':
        if prediction_type == 'cls':
            test_acc_1 = round(scores['test_acc'], 4)
            test_score_1 = round(scores['test_score'], 4)
            res = [fs_method, omics_type, k,
                   estimator, repeat, scores['pred_prob_s'],
                   scores['pred_label_s'], scores['label'],
                   test_acc_1, test_score_1]
            print(f'acc:{test_acc_1}, auroc: {test_score_1}')
        elif prediction_type == 'reg':
            test_mse_1 = round(scores['test_mse'], 4)
            test_score_1 = round(scores['test_score'], 4)
            res = [fs_method, omics_type, k,
                   estimator, repeat, scores['pred_label_s'], scores['label'],
                   test_mse_1, test_score_1]
            print(f'mse:{test_mse_1}, r2: {test_score_1}')
        elif prediction_type == 'sur':
            test_score_1 = round(scores['test_score'], 4)
            res = [fs_method, omics_type, k,
                   estimator, repeat, scores['label'], 
                   scores['test_risk_score'], test_score_1]
            print(f'c-index: {test_score_1}')
    elif mode == 'full':
        if fs_method != 'pca_ex':
            s_features = ','.join(selected_features)
        else:
            s_features = 'NA'
        if data['test_2'] is not None and data['test_2']['X'] is not None:
            if data['test_2']['y'] is not None:
                if prediction_type == 'cls':
                    test_acc_2 = round(scores['test_acc_2'], 4)
                    test_score_2 = round(scores['test_score_2'], 4)
                    res = [fs_method, omics_type, k, estimator,
                           s_features, json.dumps(cluster_membership), -1, -1,
                           scores['pred_prob_s_2'], scores['pred_label_s_2'],
                           scores['label_2'], test_acc_2, test_score_2]
                    print(f'acc:{test_acc_2}, auroc:{test_score_2}')
                elif prediction_type == 'reg':
                    test_mse_2 = round(scores['test_mse_2'], 4)
                    test_score_2 = round(scores['test_score_2'], 4)
                    res = [fs_method, omics_type, k, estimator,
                           s_features, json.dumps(cluster_membership), -1, -1,
                           scores['pred_label_s_2'],
                           scores['label_2'], test_mse_2, test_score_2]
                    print(f'mse:{test_mse_2}, r2:{test_score_2}')
                elif prediction_type == 'sur':
                    test_score_2 = round(scores['test_score_2'], 4)
                    res = [fs_method, omics_type, k, estimator,
                           s_features, json.dumps(cluster_membership), -1,
                           scores['label_2'], 
                           scores['test_risk_score_2'], test_score_2]
                    print(f'c-index:{test_score_2}')
            else:
                if prediction_type == 'cls':
                    res = [fs_method, omics_type, k, estimator,
                        s_features, json.dumps(cluster_membership), -1, -1,
                        scores['pred_prob_s_2'], scores['pred_label_s_2']]
                elif prediction_type == 'reg':
                    res = [fs_method, omics_type, k, estimator,
                        s_features, json.dumps(cluster_membership), -1, -1,
                        scores['pred_label_s_2']]
                elif prediction_type == 'sur':
                    res = [fs_method, omics_type, k, estimator,
                        s_features, json.dumps(cluster_membership), -1,
                        scores['test_risk_score_2']]
        else:
            if prediction_type == 'sur':
                res = [fs_method, omics_type, k, estimator,
                    s_features, json.dumps(cluster_membership), -1]
            else:
                res = [fs_method, omics_type, k, estimator,
                    s_features, json.dumps(cluster_membership), -1, -1]

    return res


def run_single_fs_method(data, fs_method, run_config, k, repeat, estimator,
                         mode, seed):
    n_jobs = run_config['n_jobs']
    percentile = run_config['percentile']
    prediction_type = data['desc']['prediction_type']
    est = config.get_estimator(seed, estimator, prediction_type)

    print(f'k={k}, repeat={repeat+1}, estimator={estimator},'
          f' fs_method={fs_method}', flush=True)

    # pipeline steps
    target_view_name = data['desc']['target_view_name']
    X_train_combined = None
    y_train = data['train']['y']
    X_test_combined = y_test = None
    X_test_2 = y_test_2 =  None
    view_names = data['desc']['view_names']
    for view in view_names:
        cur_train = data['train'][view]
        X_train_combined = pd.concat([X_train_combined, cur_train['X']],
                                     axis=1)
        if data['test'] is not None:
            cur_test = data['test'][view]
            X_test_combined = pd.concat([X_test_combined, cur_test['X']],
                                        axis=1)
    if data['test'] is not None:
        y_test = data['test']['y']
        y_test = get_y(y_test, prediction_type)
    if data['test_2'] is not None:
        X_test_2 = data['test_2']['X']
        y_test_2 = data['test_2']['y']
        y_test_2 = get_y(y_test_2, prediction_type)

    p_steps = []
    fs = FeatureSelector(views=view_names,
                         target_view=target_view_name,
                         method=fs_method,
                         k=k,
                         weighted=True,
                         prediction_type=prediction_type)
    # assume all data are standardized
    # it is a bit tricky to add this to the pipeline
    # because the independent test data may have
    # different number of input features, so the full model trained
    # can not be directly used to make predictions
    p_steps.append(('featureselector', fs))
    p_steps.append((estimator, est))
    cachedir = mkdtemp()
    memory = Memory(location=cachedir, verbose=0)
    pipe = Pipeline(steps=p_steps, memory=memory)
    param_grid = get_estimator_parameter_grid(estimator, prediction_type)
    param_grid['featureselector__percentile'] = percentile
    score_func = {
        'cls': 'roc_auc',
        'reg': 'r2',
         # use the default scorer of the estimator
        'sur': None
    }
    cv = {
        'cls': StratifiedKFold(3),
        'reg': KFold(3),
        'sur': KFold(3)
    }
    grid = GridSearchCV(pipe, param_grid, scoring=score_func[prediction_type],
                        cv=cv[prediction_type],
                        n_jobs=n_jobs,
                        verbose=0)
    y_train = get_y(y_train, prediction_type)
    grid.fit(X_train_combined, y_train)
    rmtree(cachedir)
    
    scores = get_scores(prediction_type, grid, X_test_combined,
                        y_test, X_test_2, y_test_2)
    res = set_up_results(data, mode, run_config, prediction_type, fs_method,
                         k, estimator, repeat, scores, grid)
    return res


def get_y(y_df, prediction_type):
    """
    convert data frame to structured array for survival type
    """
    if prediction_type == 'sur':
        col_event = y_df.columns[0]
        col_time = y_df.columns[1]
        y = np.empty(dtype=[(col_event, np.bool), (col_time, np.float64)],
                    shape=y_df.shape[0])
        y[col_event] = (y_df[col_event] == 1).values
        y[col_time] = y_df[col_time].values
    else:
        y = y_df.values.ravel()
    return y


def run_single_estimator(data, run_config, k, repeat, fs_method,
                         estimator, mode, seed):
    n_view = len(data['desc']['view_names'])
    res = []
    if fs_method is None:
        # proms and proms_mo (if there are more than 1 view available)
        cur_res = run_single_fs_method(data, 'proms', run_config, k, repeat,
                      estimator, mode, seed)
    
        res.append(cur_res)
        if n_view > 1:
            cur_res = run_single_fs_method(data, 'proms_mo', run_config, k,
                          repeat, estimator, mode, seed)
            res.append(cur_res)
        if run_config['include_pca']:
            cur_res = run_single_fs_method(data, 'pca_ex', run_config, k,
                          repeat, estimator, mode, seed)
            res.append(cur_res)
    else:
        cur_res = run_single_fs_method(data, fs_method, run_config, k, repeat,
                      estimator, mode, seed)
        res.append(cur_res)

    return res


def prepare_data(all_data, repeat, mode):
    test_ratio = 0.3
    y = all_data['y']
    prediction_type = all_data['desc']['prediction_type']
    n_view = all_data['desc']['n_view']
    dataset_name = all_data['desc']['name']
    prediction_type = all_data['desc']['prediction_type']
    view_names = all_data['desc']['view_names']
    target_view_name = all_data['desc']['target_view']
    if mode == 'eval':
        if prediction_type == config.prediction_map['classification']:
            y_train, y_test = train_test_split(y, test_size=test_ratio,
                                  stratify=y, random_state=repeat)
        else:
            y_train, y_test = train_test_split(y, test_size=test_ratio,
                                  random_state=repeat)
    else:  # full model
        y_train, y_test = y, None

    data = {'desc': {},
            'train': {},
            'test': None,
            'test_2': None
            }
    data['desc']['name'] = dataset_name
    data['desc']['view_names'] = view_names
    data['desc']['target_view_name'] = target_view_name
    data['desc']['prediction_type'] = prediction_type
    if mode == 'eval':
        data['test'] = {}
    if all_data['X_test'] is not None:
        data['test_2'] = {}

    for i in range(n_view):
        cur_view_name = view_names[i]
        data['train'][cur_view_name] = {}
        if mode == 'eval':
            data['test'][cur_view_name] = {}
        cur_X = all_data['X'][cur_view_name]
        cur_X_train = cur_X.loc[y_train.index, :]
        data['train'][cur_view_name]['X'] = cur_X_train
        if mode == 'eval':
            cur_X_test = cur_X.loc[y_test.index, :]
            data['test'][cur_view_name]['X'] = cur_X_test

        if all_data['X_test'] is not None:
            data['test_2']['X'] = all_data['X_test']
            data['test_2']['y'] = all_data['y_test']

    data['train']['y'] = y_train
    if mode == 'eval':
        data['test']['y'] = y_test

    return data


def run_single_repeat(all_data, run_config, k, repeat, fs_method,
                      mode, seed):
    data = prepare_data(all_data, repeat, mode)
    estimators = run_config['estimators']
    res = []
    for estimator in estimators:
        cur_res = run_single_estimator(data, run_config, k, repeat, fs_method,
                      estimator, mode, seed)
        res.extend(cur_res)
    return res


def run_single_k(all_data, run_config, k, fs_method, mode, seed):
    n_repeats = run_config['repeat']
    res = []
    for repeat in range(n_repeats):
        cur_res = run_single_repeat(all_data, run_config, k, repeat,
                      fs_method, mode, seed)
        res.extend(cur_res)
    return res


def get_estimator_parameter_grid(estimator, prediction_type='cls'):
    """
    get parameter grid for pipeline
    """
    pg = config.parameter_grid[prediction_type]
    if not estimator in pg:
        raise ValueError(f'estimator "{estimator}" not supported for prediction type "{prediction_type}"')
    pipeline_pg = {}
    for parameter in pg[estimator]:
        pipeline_pg[estimator + '__' + parameter] = pg[estimator][parameter]

    return pipeline_pg


def get_results_col_name(all_data, mode='eval'):
    """
    set result data frame column names
    """
    prediction_type = all_data['desc']['prediction_type']
    check_prediction_type_ok(prediction_type)

    if mode == 'eval':
        if prediction_type == 'cls':
            column_names = ['fs', 'type', 'k', 'estimator', 'repeat',
                            'val_score', 'val_pred_label',
                            'val_label', 'val_acc', 'val_auroc']
        elif prediction_type == 'reg':
            column_names = ['fs', 'type', 'k', 'estimator', 'repeat',
                            'val_pred_label', 'val_label', 'val_mse', 'val_r2']
        elif prediction_type == 'sur':
            column_names = ['fs', 'type', 'k', 'estimator', 'repeat',
                            'val_label', 'val_risk_score', 'val_c_index']
    elif mode == 'full':
        if all_data['desc']['has_test']:
            if all_data['y_test'] is not None:
                if prediction_type == 'cls':
                    column_names = ['fs', 'type', 'k', 'estimator',
                                    'features', 'membership',
                                    'mean_val_acc', 'mean_val_auroc',
                                    'test_score', 'test_pred_label',
                                    'test_label', 'test_accuracy', 'test_auroc']
                elif prediction_type == 'reg':
                    column_names = ['fs', 'type', 'k', 'estimator',
                                    'features', 'membership',
                                    'mean_val_mse', 'mean_val_r2',
                                    'test_pred_label', 'test_label',
                                    'test_mse', 'test_r2']
                elif prediction_type == 'sur':
                    column_names = ['fs', 'type', 'k', 'estimator',
                                    'features', 'membership',
                                    'mean_val_c_index',
                                    'test_label', 'test_risk_score', 'test_c_index']
            else:
                if prediction_type == 'cls':
                    column_names = ['fs', 'type', 'k', 'estimator',
                                    'features', 'membership',
                                    'mean_val_acc', 'mean_val_auroc',
                                    'test_score', 'test_pred_label']
                elif prediction_type == 'reg':
                    column_names = ['fs', 'type', 'k', 'estimator',
                                    'features', 'membership',
                                    'mean_val_mse', 'mean_val_r2',
                                    'test_pred_label']
                elif prediction_type == 'sur':
                    column_names = ['fs', 'type', 'k', 'estimator',
                                    'features', 'membership',
                                    'mean_val_c_index', 'test_risk_score']
        else:
            if prediction_type == 'cls':
                column_names = ['fs', 'type', 'k', 'estimator',
                                'features', 'membership',
                                'mean_val_acc', 'mean_val_auroc']
            elif prediction_type == 'reg':
                column_names = ['fs', 'type', 'k', 'estimator',
                                'features', 'membership',
                                'mean_val_mse', 'mean_val_r2']
            if prediction_type == 'sur':
                column_names = ['fs', 'type', 'k', 'estimator',
                                'features', 'membership',
                                'mean_val_c_index']

    return column_names


def select_for_full_model(df, prediction_type):
    """
    select best configuration to train a full model
    """
    res = {}
    if prediction_type == 'cls':
        df_sel = df[['fs', 'k', 'estimator', 'val_acc', 'val_auroc']]
        # pca_ex cannot be used for full model
        df_sel = df_sel[df_sel.fs != 'pca_ex']
        df_sel = df_sel.groupby(['fs', 'k', 'estimator']).mean()
        df_sel = df_sel.reset_index()
        best_auroc_idx = df_sel['val_auroc'].argmax()
        fs_sel = df_sel.loc[best_auroc_idx, 'fs']
        k_sel = df_sel.loc[best_auroc_idx, 'k']
        estimator_sel = df_sel.loc[best_auroc_idx, 'estimator']
        best_mean_acc = df_sel.loc[best_auroc_idx, 'val_acc']
        best_mean_auroc = df_sel.loc[best_auroc_idx, 'val_auroc']
        res['fs_sel'] = fs_sel
        res['k_sel'] = k_sel
        res['estimator_sel'] = estimator_sel
        res['best_mean_acc'] = best_mean_acc
        res['best_mean_auroc'] = best_mean_auroc
    elif prediction_type == 'reg':
        df_sel = df[['fs', 'k', 'estimator', 'val_mse', 'val_r2']]
        df_sel = df_sel[df_sel.fs != 'pca_ex']
        df_sel = df_sel.groupby(['fs', 'k', 'estimator']).mean()
        df_sel = df_sel.reset_index()
        best_auroc_idx = df_sel['val_r2'].argmax()
        fs_sel = df_sel.loc[best_auroc_idx, 'fs']
        k_sel = df_sel.loc[best_auroc_idx, 'k']
        estimator_sel = df_sel.loc[best_auroc_idx, 'estimator']
        best_mean_mse = df_sel.loc[best_auroc_idx, 'val_mse']
        best_mean_r2 = df_sel.loc[best_auroc_idx, 'val_r2']
        res['fs_sel'] = fs_sel
        res['k_sel'] = k_sel
        res['estimator_sel'] = estimator_sel
        res['best_mean_mse'] = best_mean_mse
        res['best_mean_r2'] = best_mean_r2
    elif prediction_type == 'sur':
        df_sel = df[['fs', 'k', 'estimator', 'val_c_index']]
        df_sel = df_sel[df_sel.fs != 'pca_ex']
        df_sel = df_sel.groupby(['fs', 'k', 'estimator']).mean()
        df_sel = df_sel.reset_index()
        best_auroc_idx = df_sel['val_c_index'].argmax()
        fs_sel = df_sel.loc[best_auroc_idx, 'fs']
        k_sel = df_sel.loc[best_auroc_idx, 'k']
        estimator_sel = df_sel.loc[best_auroc_idx, 'estimator']
        best_mean_c_index = df_sel.loc[best_auroc_idx, 'val_c_index']
        res['fs_sel'] = fs_sel
        res['k_sel'] = k_sel
        res['estimator_sel'] = estimator_sel
        res['best_mean_c_index'] = best_mean_c_index

    return res


def run_fs(all_data, run_config, run_version, output_root, seed):
    k = run_config['k']
    dataset_name = all_data['desc']['name']
    prediction_type = all_data['desc']['prediction_type']

    # evaluate: performance evaluation, select features with
    #           model built with train set and evaluate in validation set
    column_names = get_results_col_name(all_data, 'eval')
    out_dir_run = os.path.join(output_root, run_version)
    res = []
    for cur_k in k:
        cur_res = run_single_k(all_data, run_config, cur_k, None, 'eval', seed)
        res.extend(cur_res)

    results_df = pd.DataFrame(res, columns=column_names)
    out_file = dataset_name + '_results_'
    out_file = out_file + run_version + '_eval.tsv'
    out_file = os.path.join(out_dir_run, out_file)
    results_df.to_csv(out_file, header=True, sep='\t', index=False)

    # we will select the best combination of fs, k, estimator
    # based on cross validation results (average score) from the previous step
    # to fit a full model
    res_dict = select_for_full_model(results_df, prediction_type)

    # # full: build full model with all data and test in another
    # #       indpendent data set if available
    column_names = get_results_col_name(all_data, 'full')
    # re-define run_config file
    run_config['repeat'] = 1
    k_sel = res_dict['k_sel']
    run_config['k'] = [k_sel]
    run_config['estimators'] = [res_dict['estimator_sel']]
    fs_method = res_dict['fs_sel']

    res = []
    cur_res = run_single_k(all_data, run_config, k_sel, fs_method,
                           'full', seed)
    if prediction_type == 'cls':
        cur_res[0][6] = res_dict['best_mean_acc']
        cur_res[0][7] = res_dict['best_mean_auroc']
    elif prediction_type == 'reg':
        cur_res[0][6] = res_dict['best_mean_mse']
        cur_res[0][7] = res_dict['best_mean_r2']
    elif prediction_type == 'sur':
        cur_res[0][6] = res_dict['best_mean_c_index']

    res.extend(cur_res)
    results_df = pd.DataFrame(res, columns=column_names)
    out_file = dataset_name + '_results_'
    out_file = out_file + run_version + '_full.tsv'
    out_file = os.path.join(out_dir_run, out_file)
    results_df.to_csv(out_file, header=True, sep='\t', index=False,
                      quoting=csv.QUOTE_NONE)


def check_data_config(config_file):
    """
    verify data configuration file
    """
    with open(config_file) as config_fh:
        data_config = yaml.load(config_fh)

    required_fields = {'project_name', 'data_directory', 'train_data_directory', 'target_view',
                       'target_label', 'data'}
    allowed_fields = required_fields | {'test_data_directory'}
    if not required_fields <= data_config.keys() <= allowed_fields:
        raise Exception(f'provided fields: {sorted(data_config.keys())}\n'
                        f'config required fields: {sorted(required_fields)}\n'
                        f'allowed fields: {sorted(allowed_fields)}')

    test_dataset = data_config['test_data_directory'] if 'test_data_directory' in data_config else None
    data_required_fields = {'train'}
    if test_dataset is not None:
        data_allowed_fields = {'test'} | data_required_fields
    else:
        data_allowed_fields = data_required_fields
    data_provided_fields = data_config['data'].keys()
    if not data_required_fields <= data_provided_fields <= data_allowed_fields:
        raise Exception(f'data section provided fields: {sorted(data_provided_fields)}\n'
                        f'required fields: {sorted(data_required_fields)}\n'
                        f'allowed fileds: {sorted(data_allowed_fields)}')

    train_required_fields = {'label', 'view'}
    train_provided_fields = data_config['data']['train'].keys()
    if not train_required_fields <= train_provided_fields:
        raise Exception(f'train data required fields: {sorted(train_provided_fields)}\n'
                        f'required fields: {sorted(train_required_fields)}')


def create_dataset(config_file, output_run):
    """ create data structure from input data files """
    print(f'data config file: {config_file}')
    check_data_config(config_file)
    with open(config_file) as config_fh:
        data_config = yaml.load(config_fh)
        data_root = data_config['data_directory']
        if not os.path.isabs(data_root):
            # relative to the data config file
            config_root = os.path.abspath(os.path.dirname(config_file))
            data_root = os.path.join(config_root, data_root)
        ds_name = data_config['project_name']
    all_data = Dataset(name=ds_name, root=data_root, config_file=config_file,
                       output_dir=output_run)()
    return all_data


def check_run_config(run_config_file, n_train_sample, prediction_type):
    with open(run_config_file) as config_fh:
        run_config = yaml.load(config_fh)
        if not 'n_jobs' in run_config:
            # assume running on a node with 4 cpus
            run_config['n_jobs'] = 4
        if not 'repeat' in run_config:
            run_config['repeat'] = 5
        if not 'k' in run_config:
            raise Exception('must specifiy k in run configuration file.')
        k = run_config['k']
        k_max = sorted(k)[-1]
        if k_max > int(0.25*n_train_sample):
            raise Exception('largest k should be less than 25% '
                    'of the number of training samples')
        default_estimators = config.default_estimators[prediction_type]             
        if not 'estimators' in run_config:
            run_config['estimators'] = default_estimators
        else:
            if not set(run_config['estimators']).issubset(default_estimators):
                raise Exception(f'supported estimators:{default_estimators}')
        if not 'percentile' in run_config:
            run_config['percentile'] = [1, 5, 10, 20]
        else:
            all_ok = all(x > 0 and x < 50 for x in run_config['percentile']) 
            if not all_ok: 
                raise Exception('all percentile values must be > 0 and < 50')
    
    return run_config


def main():
    # ignore warnings from joblib
    warnings.filterwarnings('ignore')
    os.environ['PYTHONWARNINGS'] = 'ignore'
    parser = get_parser()
    args = parser.parse_args()
    run_config_file = args.run_config_file
    data_config_file = args.data_config_file
    output_root = args.output_root
    include_pca = args.include_pca
    # time stamp
    run_version = args.run_version
    # random seed for full model (if applicable)
    seed = args.seed
    # prepare output directory
    out_dir_run = os.path.join(output_root, run_version)
    out_dir_run_full = os.path.join(out_dir_run, 'full_model')
    if not os.path.exists(out_dir_run_full):
        os.makedirs(out_dir_run_full)
    all_data = create_dataset(data_config_file, out_dir_run)
    n_train_sample = all_data['y'].shape[0]
    prediction_type = all_data['desc']['prediction_type']
    run_config = check_run_config(run_config_file, n_train_sample,
                     prediction_type)
    run_config['run_version'] = run_version
    run_config['output_root'] = output_root
    run_config['include_pca'] = include_pca

    run_fs(all_data, run_config, run_version, output_root, seed)


def is_valid_file(arg):
    """ check if the file exists """
    arg = os.path.abspath(arg)
    if not os.path.exists(arg):
        msg = "The file %s does not exist!" % arg
        raise argparse.ArgumentTypeError(msg)
    else:
        return arg


def date_time_now():
    """ get the current date and time """
    now = datetime.now()
    date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    return date_time


def get_parser():
    ''' get arguments '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', dest='run_config_file',
                        type=is_valid_file,
                        required=True,
                        help='configuration file for the run',
                        metavar='FILE',
                        )
    parser.add_argument('-d', '--data', dest='data_config_file',
                        type=is_valid_file,
                        required=True,
                        help='configuration file for data set',
                        metavar='FILE',
                        )
    parser.add_argument('-s', '--seed', dest='seed',
                        default=42,
                        type=int,
                        help='random seed '
                        )
    parser.add_argument('-o', '--output', dest='output_root',
                        default='results',
                        type=str,
                        help='output directory'
                        )
    parser.add_argument('-r', '--run_version', dest='run_version',
                        default=date_time_now(),
                        type=str,
                        help='name of the run, default to current date/time'
                        )
    parser.add_argument('-p', '--include_pca', dest='include_pca',
                        default=False,
                        action='store_true',
                        help='include supervised PCA method in the results'
                       )
    return parser


if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')
    main()
