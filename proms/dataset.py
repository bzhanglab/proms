import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
import yaml
import matplotlib.pyplot as plt
import pickle
import re
from proms import config


class Data(object):
    """Base class for data set"""

    def __init__(self, name, root, config_file, output_dir):
        self.name = name
        self.root = root
        self.all_data = None
        self.output_dir = output_dir
        if output_dir is not None and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        config_file = os.path.join(root, config_file)
        with open(config_file) as fh:
            self.config = yaml.load(fh)
        self.has_test = self.has_test_set()

    def has_test_set(self):
        """ does the dataset contain independent test set """
        return 'test_data_directory' in self.config

    def load_data(self, data_file, samples, vis=False):
        # load feature data
        X = pd.read_csv(data_file, sep='\t', index_col=0)
        # remove features with missing values
        X.dropna(axis=0, inplace=True)

        # if there are duplicated rows (genes), take the row average
        X = X.groupby(X.index).mean()
        # sample x features
        X = X.T
        if samples is not None:
            X = X.loc[samples, :]
        if vis and self.output_dir is not None:
            myfig = plt.figure()
            X_temp = X.T
            flierprops = dict(marker='o', markersize=1)
            X_temp.boxplot(grid=False, rot=90, fontsize=2,
                flierprops=flierprops)
            upper_dir = os.path.basename(os.path.dirname(data_file))
            file_name = (self.name + '_' + upper_dir + '_' +
                         os.path.basename(data_file) + '.pdf')
            myfig.savefig(os.path.join(self.output_dir, file_name),
                          format='pdf')

        # X.sort_index(inplace=True)
        return X

    def save_data(self):
        """save all data into a file"""
        if self.output_dir is not None:
            fname = self.name + '_all_data.pkl'
            with open(os.path.join(self.output_dir, fname), 'wb') as fh:
                pickle.dump(self.all_data, fh, protocol=pickle.HIGHEST_PROTOCOL)


class Dataset(Data):
    """ data set"""
    def __init__(self, name, root, config_file, output_dir=None):
        self.clin_data = None
        self.prediction_type = None
        # only valid for classification
        self.classes = None
        super().__init__(name, root, config_file, output_dir)

    def check_prediction_type(self):
        """
        Based on the target data, infer if it is a
        regression, classification or survival analysis task
        """
        if self.clin_data is None:
            raise ValueError('No clinical data has been set')

        target_vals = self.clin_data.iloc[:,0].values
        target_dtype = target_vals.dtype
        # for classification, binary only
        if target_dtype == np.int64:
            uniq_len = len(np.unique(target_vals))
            if uniq_len == 2:
                return config.prediction_map['classification']
            if uniq_len > 2:
                return config.prediction_map['regression']
            raise ValueError('all target values are the same')

        if target_dtype == np.float64:
            return config.prediction_map['regression']

        if target_dtype == np.object:
            target_vals_str = target_vals.astype(str)
            r_surv = re.compile('.+,.+')
            if all(r_surv.match(item) for item in list(target_vals_str)):
                return config.prediction_map['survival']
            uniq_len = len(np.unique(target_vals_str))
            if uniq_len == 2:
                return config.prediction_map['classification']
            if uniq_len > 2:
                raise ValueError('multiclass classification not supported')
            raise ValueError('all target values are the same')

        raise ValueError('wrong target values')

    def load_clin_surv(self, clin_file, target_label):
        """
        load y for survival analysis
        """
        clin_data = pd.read_csv(clin_file, sep='\t', index_col=0)
        col_event = target_label[0]
        col_time = target_label[1]
        y = np.empty(dtype=[(col_event, np.bool), (col_time, np.float64)],
                     shape=clin_data.shape[0])
        y[col_event] = (clin_data[col_event] == 1).values
        y[col_time] = clin_data[col_time].values
        sample_names = clin_data.index.values
        y = pd.DataFrame.from_records(y, index=sample_names)
        return y

    def __call__(self):
        print('processing data ...')
        train_dataset = self.config['train_data_directory']
        target_view = self.config['target_view']
        target_label = self.config['target_label']
        clin_file = self.config['data']['train']['label']['file']
        clin_file = os.path.join(self.root, train_dataset, clin_file)
        if isinstance(target_label, list):  # survival
            clin_data = self.load_clin_surv(clin_file, target_label)
            train_sample = clin_data.index
            self.clin_data = clin_data
            self.prediction_type = config.prediction_map['survival']
        else:
            # the first col will be the index
            clin_data = pd.read_csv(clin_file, sep='\t', index_col=0)
            # samples X phenotype
            clin_data = clin_data.loc[:, [target_label]]
            self.clin_data = clin_data
            self.prediction_type = self.check_prediction_type()
            train_sample = clin_data.index
        print(f'prediction type: {self.prediction_type}')
        # for now: must be two-class classification
        if self.prediction_type == config.prediction_map['classification']:
            le = preprocessing.LabelEncoder()
            clin_vals = clin_data.iloc[:, 0].values
            le.fit(clin_vals)
            self.classes = le.classes_
            clin_data.iloc[:, 0] = le.transform(clin_vals)
            self.clin_data = clin_data

        y = clin_data

        # get the name of samples that have all the omics data available
        all_views = self.config['data']['train']['view']
        n_views = len(all_views)
        all_view_names = [item['type'] for item in all_views]
        all_view_names.remove(target_view)
        all_view_ordered = [target_view] + sorted(all_view_names)

        if self.has_test:
            test_dataset = self.config['test_data_directory']
            all_test_views = self.config['data']['test']['view']
            if 'label' in self.config['data']['test']:
                test_clin_file = self.config['data']['test']['label']['file']
                test_clin_file = os.path.join(self.root, test_dataset, test_clin_file)
                if self.prediction_type == 'sur':
                    test_clin_data = self.load_clin_surv( test_clin_file, target_label)
                    test_samples = test_clin_data.index
                else:
                    test_clin_data = pd.read_csv(test_clin_file, sep='\t',
                                                 index_col=0)
                    test_samples = test_clin_data.index
                    # samples X phenotype
                    test_clin_data = test_clin_data.loc[:, [target_label]]
                if self.prediction_type == config.prediction_map['classification']:
                    le = preprocessing.LabelEncoder()
                    test_clin_vals = test_clin_data.iloc[:, 0].values
                    le.fit(test_clin_vals)
                    if any(le.classes_ != self.classes):
                        raise ValueError('class label in test dataset not'
                            ' matching training data')
                    test_clin_data.iloc[:, 0] = le.transform(test_clin_vals)
                y_final_test_2 = test_clin_data
            else:
                # no label provided, only for prediction
                test_samples = None
                y_final_test_2 = None
            selected_view = filter(lambda view: view['type'] == target_view,
                                   all_test_views)
            test_view_file = list(selected_view)[0]['file']
            test_view_file = os.path.join(self.root, test_dataset, test_view_file)
            X_final_test_2 = self.load_data(test_view_file, test_samples)
        else:
            X_final_test_2 = None
            y_final_test_2 = None

        all_data_ = {'desc': {
                        'prediction_type': self.prediction_type,
                        'name': self.name,
                        'has_test': self.has_test,
                        'target_label': target_label,
                        'target_view': target_view,
                        'n_view': n_views,
                        'view_names': all_view_ordered
                        },
                     'X': {},
                     'y': y,
                     'X_test': None,
                     'y_test': y_final_test_2
                    }

        for i in range(n_views):
            view_name = all_views[i]['type']
            print(f'view: {view_name}')
            view_file = os.path.join(self.root, train_dataset,
                                     all_views[i]['file'])
            X = self.load_data(view_file, train_sample)
            if view_name == target_view:
                if self.has_test:
                    # find out common features in the two data
                    # sets for the target view
                    common_features = sorted(list(set(X.columns.values) & set(
                        X_final_test_2.columns.values)))
                    print('train data target view features length {}'.format(
                        len(X.columns.values)))
                    print('test data target view features length {}'.format(
                        len(X_final_test_2.columns.values)))
                    print('common target view features length {}'.format(
                        len(common_features)))
                    X = X[common_features]
                    X_final_test_2 = X_final_test_2[common_features]
                    X_final_test_2 = X_final_test_2.add_prefix(view_name + '_')

            X = X.add_prefix(view_name + '_')
            all_data_['X'][view_name] = X

        if self.has_test:
            all_data_['X_test'] = X_final_test_2

        print('number of samples: {}'.format(all_data_['y'].shape[0]))

        if self.has_test:
            all_data_['X_test'] = X_final_test_2
            if all_data_['y_test'] is not None:
                print('test sample size: {}'.format(all_data_['y_test'].shape[0]))
            else:
                print('test sample size: {}'.format(all_data_['X_test'].shape[0]))

        self.all_data = all_data_
        # self.save_data()
        return all_data_
