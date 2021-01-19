import multiprocessing
import os
import json
import csv
from datetime import datetime
from tempfile import mkdtemp
from shutil import rmtree
from numpy.lib.npyio import save
import pandas as pd
import argparse
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from joblib import Memory
from proms import Dataset, FeatureSelector


def run_single_fs_method(data, fs_method, run_config, k, repeat, estimator,
                         mode, seed):
    n_jobs = run_config['n_jobs']
    percentile = run_config['percentile']

    classifier_dict = {
        'xgboost': XGBClassifier(random_state=seed),
        'rf': RandomForestClassifier(random_state=seed),
        'svm': SVC(kernel='rbf', probability=True, random_state=seed),
        'logreg': LogisticRegression(max_iter=1000, random_state=seed)
    }
    classifier = classifier_dict[estimator]
    print(f'k={k}, repeat={repeat+1}, estimator={estimator}, fs_method={fs_method}', 
          flush=True)

    # pipeline steps
    target_view_name = data['desc']['target_view_name']
    X_train_combined = None
    y_train = data['train']['y']
    X_test_combined = None
    X_test_2 = None
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
    if data['test_2'] is not None:
        X_test_2 = data['test_2']['X']
        y_test_2 = data['test_2']['y']

    p_steps = []
    fs = FeatureSelector(views=view_names,
                         target_view=target_view_name,
                         method=fs_method,
                         k=k,
                         weighted=True)
    p_steps.append(('featureselector', fs))
    p_steps.append((estimator, classifier))
    cachedir = mkdtemp()
    memory = Memory(location=cachedir, verbose=0)
    pipe = Pipeline(steps=p_steps, memory=memory)
    param_grid = get_estimator_parameter_grid(estimator)
    param_grid['featureselector__percentile'] = percentile
    grid = GridSearchCV(pipe, param_grid, scoring='roc_auc',
                        cv=StratifiedKFold(3),
                        n_jobs=n_jobs,
                        verbose=1)
    grid.fit(X_train_combined, y_train.values.ravel())
    rmtree(cachedir)

    # evaluate on the test data set (if available)
    # and independent test set (if available)
    test_acc = test_score = 'NA'
    label = pred_label_s = pred_prob_s = 'NA'
    test_acc_2 = test_score_2 = 'NA'
    label_2 = pred_label_s_2 = pred_prob_s_2 = 'NA'
    if X_test_combined is not None:
        label = ','.join(map(str, y_test.to_numpy().squeeze()))
        pred_prob = grid.predict_proba(X_test_combined)[:, 1]
        pred_label = [1 if x >= 0.5 else 0 for x in pred_prob]
        test_score = grid.score(X_test_combined, y_test.values.ravel())
        pred_prob_s = ','.join(map('{:.4f}'.format, pred_prob))
        pred_label_s = ','.join(map(str, pred_label))
        test_acc = accuracy_score(y_test.to_numpy().squeeze(), pred_label)

    if X_test_2 is not None:
        pred_prob_2 = grid.predict_proba(X_test_2)[:, 1]
        pred_label_2 = [1 if x >= 0.5 else 0 for x in pred_prob_2]
        pred_prob_s_2 = ','.join(map('{:.4f}'.format, pred_prob_2))
        pred_label_s_2 = ','.join(map(str, pred_label_2))
        if y_test_2 is not None:
            label_2 = ','.join(map(str, y_test_2.to_numpy().squeeze()))
            test_score_2 = round(grid.score(X_test_2, y_test_2.values.ravel()), 4)
            test_acc_2 = round(accuracy_score(y_test_2.to_numpy().squeeze(),
                            pred_label_2), 4)

    # best_percentile = grid.best_params_['featureselector__percentile']
    best_fs = grid.best_estimator_.named_steps['featureselector']
    selected_features = best_fs.get_feature_names()

    if mode == 'full':
        cluster_membership = best_fs.get_cluster_membership()
        run_version = run_config['run_version']
        output_root = run_config['output_root']
        out_dir_run = os.path.join(output_root, run_version)
        out_dir_run_full = os.path.join(out_dir_run, 'full_model')
        with open(os.path.join(out_dir_run_full, 'full_model.pkl'), 'wb') as fh:
            pickle.dump(grid.best_estimator_, fh, protocol=pickle.HIGHEST_PROTOCOL)

    omics_type = 'so' if fs_method == 'proms' else 'mo'
    if mode == 'eval':
        res = [fs_method, omics_type, k,
               estimator, repeat, pred_prob_s,
               pred_label_s, label, round(test_acc, 4), round(test_score, 4)]
        print(f'acc:{round(test_acc, 4)}, auroc:{round(test_score, 4)}')
        return res
    elif mode == 'full':
        s_features = ','.join(selected_features)
        if data['test_2']['X'] is not None:
            if data['test_2']['y'] is not None:
                res = [fs_method, omics_type, k, estimator,
                    s_features, json.dumps(cluster_membership), -1, -1,
                    pred_prob_s_2, pred_label_s_2,
                    label_2, test_acc_2, test_score_2]
                print(f'acc:{test_acc_2}, auroc:{test_score_2}')
            else:
                res = [fs_method, omics_type, k, estimator,
                    s_features, json.dumps(cluster_membership), -1, -1,
                    pred_prob_s_2, pred_label_s_2]
        else:
            res = [fs_method, omics_type, k, estimator,
                s_features, json.dumps(cluster_membership), -1, -1]

        return res


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
    else:
        cur_res = run_single_fs_method(data, fs_method, run_config, k, repeat,
                      estimator, mode, seed)
        res.append(cur_res)

    return res


def prepare_data(all_data, repeat, mode):
    test_ratio = 0.3
    y = all_data['y']
    n_view = all_data['desc']['n_view']
    dataset_name = all_data['desc']['name']
    view_names = all_data['desc']['view_names']
    target_view_name = all_data['desc']['target_view']
    if mode == 'eval':
        y_train, y_test = train_test_split(y, test_size=test_ratio,
                                           stratify=y, random_state=repeat)
    else:
        y_train, y_test = y, None

    data = {'desc': {},
            'train': {},
            'test': None,
            'test_2': None
            }
    data['desc']['name'] = dataset_name
    data['desc']['view_names'] = view_names
    data['desc']['target_view_name'] = target_view_name
    if mode == 'eval':
        data['test'] = {}
    if 'X_test' in all_data:
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

        if 'X_test' in all_data:
            data['test_2']['X'] = all_data['X_test']
            data['test_2']['y'] = all_data['y_test']

    data['train']['y'] = y_train
    if mode == 'eval':
        data['test']['y'] = y_test

    return data


def run_single_repeat(all_data, run_config, k, repeat, fs_method,
                      mode, seed):
    data = prepare_data(all_data, repeat, mode)
    classifiers = run_config['classifiers']
    res = []
    for estimator in classifiers:
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


def get_estimator_parameter_grid(classifier):
    if classifier == 'xgboost':
        param_grid = {classifier + '__max_depth': [2, 4, 6, 8, 10],
                      classifier + '__n_estimators': range(50, 400, 50)
                      }
    elif classifier == 'rf':
        param_grid = {classifier + '__max_depth': [2, 4, 6, 8, 10],
                      classifier + '__n_estimators': range(50, 400, 50)
                      }
    elif classifier == 'svm':
        param_grid = {classifier + '__C': [0.001, 0.01, 0.1, 1, 10, 100],
                      classifier + '__gamma': [0.001, 0.01, 0.1, 1, 10, 100]
                      }
    elif classifier == 'logreg':
        param_grid = {classifier + '__C': [0.001, 0.01, 0.1, 1, 10, 100],
                      }
    else:
        raise ValueError(f'classifier {classifier} not supported')
    return param_grid


def run_fs(all_data, run_config, run_version, output_root, seed):
    k = run_config['k']
    dataset_name = all_data['desc']['name']

    # evaluate: performance evaluation, select features with
    #           model built with train set and evaluate in validation set
    column_names = ['fs', 'type', 'k', 'classifier', 'repeat',
                    'val_score', 'val_pred_label',
                    'val_label', 'val_acc', 'val_auroc']
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

    # we will select the best combination of fs, k, classifier
    # based on cross validation results (average aurco) from the previous step
    # to fit a full model
   
    results_df_sel = results_df[['fs', 'k', 'classifier', 'val_acc',
                                'val_auroc']]
    results_df_sel = results_df_sel.groupby(['fs', 'k', 'classifier']).mean()
    results_df_sel = results_df_sel.reset_index()
    best_auroc_idx = results_df_sel['val_auroc'].argmax()
    fs_sel = results_df_sel.loc[best_auroc_idx, 'fs']
    k_sel = results_df_sel.loc[best_auroc_idx, 'k']
    classifier_sel = results_df_sel.loc[best_auroc_idx, 'classifier']
    best_avg_acc = results_df_sel.loc[best_auroc_idx, 'val_acc']
    best_avg_auroc = results_df_sel.loc[best_auroc_idx, 'val_auroc']

    # # full: build full model with all data and test in another
    # #       indpendent data set if available
    if all_data['desc']['has_test']:
        if all_data['y_test'] is not None:
            column_names = ['fs', 'type', 'k', 'classifier',
                            'features', 'membership',
                            'avg_val_acc', 'avg_val_auroc',
                            'test_score', 'test_pred_label', 'test_label',
                            'test_accuracy', 'test_auroc']
        else:
            column_names = ['fs', 'type', 'k', 'classifier',
                            'features', 'membership',
                            'avg_val_acc', 'avg_val_auroc',
                            'test_score', 'test_pred_label']
    else:
        column_names = ['fs', 'type', 'k', 'classifier',
                        'features', 'membership',
                        'avg_val_acc', 'avg_val_auroc']
    # re-define run_config file
    run_config['repeat'] = 1
    run_config['k'] = [k_sel]
    run_config['classifiers'] = [classifier_sel]
    fs_method = fs_sel

    res = []
    cur_res = run_single_k(all_data, run_config, k_sel, fs_method,
                           'full', seed)
    cur_res[0][6] = best_avg_acc
    cur_res[0][7] = best_avg_auroc
    res.extend(cur_res)
    results_df = pd.DataFrame(res, columns=column_names)
    out_file = dataset_name + '_results_'
    out_file = out_file + run_version + '_full.tsv'
    out_file = os.path.join(out_dir_run, out_file)
    results_df.to_csv(out_file, header=True, sep='\t', index=False, quoting=csv.QUOTE_NONE)


def check_data_config(config_file):
    with open(config_file) as config_fh:
        data_config = json.load(config_fh)

    required_fields = {'name', 'data_root', 'train_dataset', 'target_view',
                       'target_label', 'data'}
    allowed_fields = {'test_dataset'} | required_fields
    if not required_fields <= data_config.keys() <= allowed_fields:
        raise Exception(f'config required fields: {required_fields}, allowed fields: {allowed_fields}')

    data_required_fields = {'train'}
    data_allowed_fields = {'test'} | data_required_fields
    if not data_required_fields <= data_config['data'].keys() <= data_allowed_fields:
        raise Exception(f'data required fields: {data_required_fields}, allowed fileds: {data_allowed_fields}')

    train_required_fields = {'label', 'view'}
    if not train_required_fields <= data_config['data']['train'].keys():
        raise Exception(f'train data required fields: {train_required_fields}')


def create_dataset(config_file, output_run):
    """ create data structure from input data files """
    print(f'data config file: {config_file}')
    check_data_config(config_file)
    with open(config_file) as config_fh:
        data_config = json.load(config_fh)
        data_root = data_config['data_root']
        if not os.path.isabs(data_root):
            # relative to the data config file
            config_root = os.path.abspath(os.path.dirname(config_file))
            data_root = os.path.join(config_root, data_root)
        ds_name = data_config['name']
    all_data = Dataset(name=ds_name, root=data_root, config_file=config_file,
                       output_dir=output_run)()
    return all_data


def check_run_config(run_config_file, n_train_sample):
    with open(run_config_file) as config_fh:
        run_config = json.load(config_fh)
        if not 'n_jobs' in run_config:
            # assume running on a node with 4 cpus
            run_config['n_jobs'] = 4
        if not 'repeat' in run_config:
            run_config['repeat'] = 5
        if not 'k' in run_config:
            raise Exception('must specifiy k in run configuration file.')
        else:
            k = run_config['k']
            k_max = sorted(k)[-1]
            if k_max > int(0.25*n_train_sample):
                raise Exception('largest k should be less than 25% of the number of training samples')
        default_classifiers = ['logreg', 'rf', 'svm', 'xgboost']
        if not 'classifiers' in run_config:
            run_config['classifiers'] = default_classifiers
        else:
            if not set(run_config['classifiers']).issubset(default_classifiers):
                raise Exception(f'supported classifiers:{default_classifiers}')
        if not 'percentile' in run_config:
            run_config['percentile'] = [1, 5, 10, 20]
        else:
            all_ok = all(x > 0 and x < 50 for x in run_config['percentile']) 
            if not all_ok: 
                raise Exception('all percentile values must be > 0 and < 50')
    
    return run_config


def main():
    parser = get_parser()
    args = parser.parse_args()
    run_config_file = args.run_config_file
    data_config_file = args.data_config_file
    output_root = args.output_root
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
    run_config = check_run_config(run_config_file, n_train_sample)
    run_config['run_version'] = run_version
    run_config['output_root'] = output_root

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
    return parser


if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')
    main()
