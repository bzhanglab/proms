import os
import sys
import re
import tempfile
import yaml
import pandas as pd
import argparse
import pickle
from proms import Data

class Dataset(Data):
    """ data set"""

    def __call__(self):
        print('processing data ...')
        predict_dataset = self.config['predict_dataset']

        # get the name of samples that have all the omics data available
        all_views = self.config['data'][predict_dataset]['view']
        n_views = len(all_views)
        all_view_names = [item['type'] for item in all_views]

        all_data_ = {'desc': {
                        'name': self.name,
                        'n_view': n_views,
                        'view_names': all_view_names
                        },
                     'X': {}
                    }

        for i in range(n_views):
            view_name = all_views[i]['type']
            print(f'view: {view_name}')
            view_file = os.path.join(self.root, 
                                     all_views[i]['file'])
            X = self.load_data(view_file, None, vis=False)

            X = X.add_prefix(view_name + '_')
            all_data_['X'][view_name] = X

        print('number of samples: {}'.format(all_data_['X'][view_name].shape[0]))

        self.all_data = all_data_
        return all_data_


def verify_features(saved_model, target_view, predict_data_file):
    selected_features = saved_model[0].selected_features
    # remove prefix 
    s_features = [re.sub(r'^' + target_view + r'_', '', i) for i in selected_features]
    X = pd.read_csv(predict_data_file, sep='\t', index_col=0)
    provided = list(X.index)
    missing_features = list(set(s_features).difference(set(provided)))
    ret_status = True if len(missing_features) == 0 else False
    return s_features, missing_features, ret_status


def prepare_data(all_data):
    n_view = all_data['desc']['n_view']
    dataset_name = all_data['desc']['name']
    view_names = all_data['desc']['view_names']

    data = {'desc': {'view_names': view_names},
            'train': {}
            }
    data['desc']['name'] = dataset_name

    for i in range(n_view):
        cur_view_name = view_names[i]
        data['train'][cur_view_name] = {}
        cur_X = all_data['X'][cur_view_name]
        data['train'][cur_view_name]['X'] = cur_X

    data['train']['y'] = None

    return data


def create_dataset(config_file, output_run=None):
    """ create data structure from input data files """
    with open(config_file) as config_fh:
        data_config = yaml.load(config_fh)
        data_root = data_config['data_root']
        if not os.path.isabs(data_root):
            # relative to the data config file
            config_root = os.path.abspath(os.path.dirname(config_file))
            data_root = os.path.join(config_root, data_root)
        ds_name = data_config['name']
    all_data = Dataset(name=ds_name, root=data_root, config_file=config_file,
                       output_dir=output_run)()
    return all_data


def main():
    parser = get_parser()
    args = parser.parse_args()
    model_file = args.saved_model_file
    data_file = args.data_file
    output_file = args.output_file

    with open(model_file, 'rb') as m_fh:
        saved_model = pickle.load(m_fh)

    target_view = saved_model[0].target_view
    prediction_type = saved_model[0].prediction_type
    # required features not provided
    required_features, missing_features, ret_status = \
        verify_features(saved_model, target_view, data_file)
    if not ret_status:
        print('selected features not provided in data file')
        print(f'required features: {required_features}')
        print(f'missing features: {missing_features}')
        sys.exit(-1)

    # create a config file internally
    # user only need to provide a tsv data file
    temp = tempfile.NamedTemporaryFile(suffix='.yml')
    config_dict = {
        'name': 'predict_dataset',
        'data_root': os.path.dirname(data_file),
        'predict_dataset': 'predict',
        'data': {
            'predict': {
                'view': [
                    {
                        'type': target_view,
                        'file': os.path.basename(data_file)
                    }
                ]
            }
        }
    }

    with open(temp.name, 'w') as fh:
        yaml.dump(config_dict, fh)

    all_data = create_dataset(temp.name)
    data = prepare_data(all_data)
    temp.close()
    X_train_combined = None
    view_names = data['desc']['view_names']
    for view in view_names:
        cur_train = data['train'][view]
        X_train_combined = pd.concat([X_train_combined, cur_train['X']],
                                     axis=1)
    if prediction_type == 'cls':
        pred_prob = saved_model.predict_proba(X_train_combined)[:,1]
        pred_label = [1 if x >= 0.5 else 0 for x in pred_prob]
        res = pd.DataFrame({'probability': pred_prob, 'label': pred_label},
                             index=X_train_combined.index)
    elif prediction_type == 'reg':
        pred_val = saved_model.predict(X_train_combined)
        res = pd.DataFrame({'label': pred_val},
                             index=X_train_combined.index)
    elif prediction_type == 'sur':
        pred_val = saved_model.predict(X_train_combined)
        res = pd.DataFrame({'risk_score': pred_val},
                             index=X_train_combined.index)
    else:
        raise ValueError(f'prediction type {prediction_type} not supported')

    res.to_csv(output_file, sep='\t', index_label='sample')
    print(f'{output_file} created.')


def is_valid_file(arg):
    """ check if the file exists """
    arg = os.path.abspath(arg)
    if not os.path.exists(arg):
        msg = "The file %s does not exist!" % arg
        raise argparse.ArgumentTypeError(msg)
    else:
        return arg


def get_parser():
    ''' get arguments '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', dest='saved_model_file',
                        type=is_valid_file,
                        required=True,
                        help='saved model pickle file',
                        metavar='FILE'
                        )
    parser.add_argument('-d', '--data', dest='data_file',
                        type=is_valid_file,
                        required=True,
                        help='tsv data file',
                        metavar='FILE'
                        )
    parser.add_argument('-o', '--output', dest='output_file',
                        required=False,
                        default='prediction_output.txt',
                        help='output file for prediction results',
                        metavar='FILE'
                        )
    return parser


if __name__ == '__main__':
    main()
