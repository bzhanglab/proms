# ProMS: protein marker selection using proteomics or multi-omics data

## Contents

- [ProMS: protein marker selection using proteomics or multi-omics data](#proms-protein-marker-selection-using-proteomics-or-multi-omics-data)
  - [Installation](#installation)
  - [Algorithms](#algorithms)
    - [ProMS: Protein marker selection with proteomics data alone](#proms-protein-marker-selection-with-proteomics-data-alone)
    - [ProMS_mo: Protein marker selection with multi-omics data](#proms_mo-protein-marker-selection-with-multi-omics-data)
  - [Data preparation](#data-preparation)
    - [Training data](#training-data)
    - [Test data](#test-data)
  - [Configuration files](#configuration-files)
  - [How to run](#how-to-run)
    - [Training/validation/test](#trainingvalidationtest)
  - [Output files](#output-files)
  - [Prediction on new data set](#prediction-on-new-data-set)
  - [Example](#example)


## Installation

To install `proms`, run:

```console
> pip install proms
```

or

```console
> pip install git+https://github.com/bzhanglab/proms
```

## Algorithms

We provide two methods for selecting protein markers. 

### ProMS: Protein marker selection with proteomics data alone

The algorithm `ProMS` (Protein Marker Selection) works as follows. As a first step to remove uninformative features, `ProMS` examines each feature individually to determine the strength of the relationship between the feature and the target variable. A symmetric AUROC score <img src="https://raw.githubusercontent.com/bzhanglab/proms/main/docs/svgs/6d4e730d903f26cecfe5284b224077c5.svg?invert_in_darkmode" align=middle width=62.042412299999995pt height=22.465723500000017pt/> is defined to evaluate such strength: <img src="https://raw.githubusercontent.com/bzhanglab/proms/main/docs/svgs/cbd5eb655b77865cb6ac468c8b6daf0c.svg?invert_in_darkmode" align=middle width=201.58999079999998pt height=24.65753399999998pt/> .

`ProMS` only keeps the features with the top <img src="https://raw.githubusercontent.com/bzhanglab/proms/main/docs/svgs/42b7c0743f6690133133222f0462f5ca.svg?invert_in_darkmode" align=middle width=24.27516959999999pt height=24.65753399999998pt/> highest <img src="https://raw.githubusercontent.com/bzhanglab/proms/main/docs/svgs/6d4e730d903f26cecfe5284b224077c5.svg?invert_in_darkmode" align=middle width=62.042412299999995pt height=22.465723500000017pt/> scores. Here <img src="https://raw.githubusercontent.com/bzhanglab/proms/main/docs/svgs/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode" align=middle width=10.57650494999999pt height=14.15524440000002pt/> is a hyper-parameter that needs to be tuned jointly with other hyperparameters of the final classifier. After the filtering step, data matrix <img src="https://raw.githubusercontent.com/bzhanglab/proms/main/docs/svgs/17104becada06c6cda0447c33ec6c846.svg?invert_in_darkmode" align=middle width=14.49764249999999pt height=22.55708729999998pt/> is reduced to <img src="https://raw.githubusercontent.com/bzhanglab/proms/main/docs/svgs/49a96e890ec991ba643ea6bcbd390091.svg?invert_in_darkmode" align=middle width=18.287603399999988pt height=24.7161288pt/> of size <img src="https://raw.githubusercontent.com/bzhanglab/proms/main/docs/svgs/a8e7abb5ab02a0c56e63b52bdea5917d.svg?invert_in_darkmode" align=middle width=42.018596399999986pt height=24.7161288pt/> where  <img src="https://raw.githubusercontent.com/bzhanglab/proms/main/docs/svgs/038e818044681e2d2b74874dd4909d3a.svg?invert_in_darkmode" align=middle width=46.72360934999999pt height=24.7161288pt/>. To further reduce the redundancy among the remaining features, `ProMS` groups <img src="https://raw.githubusercontent.com/bzhanglab/proms/main/docs/svgs/4ae3393b40dfbbbc0932cf55cbc55bc3.svg?invert_in_darkmode" align=middle width=12.060528149999989pt height=24.7161288pt/> features into <img src="https://raw.githubusercontent.com/bzhanglab/proms/main/docs/svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/> clusters with weighted k-medoids clustering in sample space. The <img src="https://raw.githubusercontent.com/bzhanglab/proms/main/docs/svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/> medoids from each cluster are selected as markers. The whole process is illustrated in the following diagram:

<center><img src="https://raw.githubusercontent.com/bzhanglab/proms/main/docs/proms.png" alt="proms" height="800"/></center>


### ProMS_mo: Protein marker selection with multi-omics data
We have <img src="https://raw.githubusercontent.com/bzhanglab/proms/main/docs/svgs/7b9a0316a2fcd7f01cfd556eedf72e96.svg?invert_in_darkmode" align=middle width=14.99998994999999pt height=22.465723500000017pt/> data sources, <img src="https://raw.githubusercontent.com/bzhanglab/proms/main/docs/svgs/47f62d38a583b29d95d9baba60567584.svg?invert_in_darkmode" align=middle width=76.32634349999998pt height=22.55708729999998pt/>, representing <img src="https://raw.githubusercontent.com/bzhanglab/proms/main/docs/svgs/7b9a0316a2fcd7f01cfd556eedf72e96.svg?invert_in_darkmode" align=middle width=14.99998994999999pt height=22.465723500000017pt/> different types of omics measurements that jointly depicts the same set of samples <img src="https://raw.githubusercontent.com/bzhanglab/proms/main/docs/svgs/4ee1de2974f920dee2a9b2b40012e488.svg?invert_in_darkmode" align=middle width=59.221880849999984pt height=14.15524440000002pt/>. <img src="https://raw.githubusercontent.com/bzhanglab/proms/main/docs/svgs/4bc7c685b27caee1f740df61469bb5a3.svg?invert_in_darkmode" align=middle width=97.25457884999999pt height=24.65753399999998pt/> is a matrix of size  <img src="https://raw.githubusercontent.com/bzhanglab/proms/main/docs/svgs/bbef302f389d7bd8e7314d934215920d.svg?invert_in_darkmode" align=middle width=42.879533399999985pt height=19.1781018pt/> where rows correspond to samples and columns correspond to features in <img src="https://raw.githubusercontent.com/bzhanglab/proms/main/docs/svgs/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode" align=middle width=5.663225699999989pt height=21.68300969999999pt/>th data source. Without the loss of generality, we use <img src="https://raw.githubusercontent.com/bzhanglab/proms/main/docs/svgs/9fa8482e85abf0b11fd06a0c5848ff1c.svg?invert_in_darkmode" align=middle width=21.050188499999987pt height=22.55708729999998pt/> to represent the proteomics data from which we seek to select a set of informative markers that can be used to predict the target labels. Similar to `ProMS`, the first step of `ProMS_mo` involves filtering out insignificant features from each data source separately. Again we use <img src="https://raw.githubusercontent.com/bzhanglab/proms/main/docs/svgs/6d4e730d903f26cecfe5284b224077c5.svg?invert_in_darkmode" align=middle width=62.042412299999995pt height=22.465723500000017pt/>. `ProMS_mo` first applies the univariate filtering to target data source <img src="https://raw.githubusercontent.com/bzhanglab/proms/main/docs/svgs/9fa8482e85abf0b11fd06a0c5848ff1c.svg?invert_in_darkmode" align=middle width=21.050188499999987pt height=22.55708729999998pt/> and keeps only the top <img src="https://raw.githubusercontent.com/bzhanglab/proms/main/docs/svgs/42b7c0743f6690133133222f0462f5ca.svg?invert_in_darkmode" align=middle width=24.27516959999999pt height=24.65753399999998pt/> features with the highest scores. We denote the minimal score among these remaining features as <img src="https://raw.githubusercontent.com/bzhanglab/proms/main/docs/svgs/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode" align=middle width=8.17352744999999pt height=22.831056599999986pt/>. For other data source, `ProMS_mo` only keeps those features with score larger than <img src="https://raw.githubusercontent.com/bzhanglab/proms/main/docs/svgs/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode" align=middle width=8.17352744999999pt height=22.831056599999986pt/>. Filtered data matrices are combined into a new matrix <img src="https://raw.githubusercontent.com/bzhanglab/proms/main/docs/svgs/084c1922f77c103e97871103b78b4416.svg?invert_in_darkmode" align=middle width=18.287603399999988pt height=24.7161288pt/> of size <img src="https://raw.githubusercontent.com/bzhanglab/proms/main/docs/svgs/a8e7abb5ab02a0c56e63b52bdea5917d.svg?invert_in_darkmode" align=middle width=42.018596399999986pt height=24.7161288pt/>, where <img src="https://raw.githubusercontent.com/bzhanglab/proms/main/docs/svgs/e4a94d5a6c99ce374d7b1434b0021cb6.svg?invert_in_darkmode" align=middle width=89.92955564999998pt height=32.256008400000006pt/> and <img src="https://raw.githubusercontent.com/bzhanglab/proms/main/docs/svgs/35389bc6640e0f68bd4e60472c572da0.svg?invert_in_darkmode" align=middle width=12.92146679999999pt height=24.7161288pt/> is the number of features in the filtered data source <img src="https://raw.githubusercontent.com/bzhanglab/proms/main/docs/svgs/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode" align=middle width=5.663225699999989pt height=21.68300969999999pt/>. Finally, weighted k-medoids clustering is performed to partition the <img src="https://raw.githubusercontent.com/bzhanglab/proms/main/docs/svgs/4ae3393b40dfbbbc0932cf55cbc55bc3.svg?invert_in_darkmode" align=middle width=12.060528149999989pt height=24.7161288pt/> features into <img src="https://raw.githubusercontent.com/bzhanglab/proms/main/docs/svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/> clusters in sample spaces. To guarantee that only protein markers are selected as medoids, `ProMS_mo` first initializes the <img src="https://raw.githubusercontent.com/bzhanglab/proms/main/docs/svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/> medoids to protein markers. During the iterative steps of optimization, a medoid can only be replaced by another protein marker if such exchange improves the objective function. After the iterative process converges, <img src="https://raw.githubusercontent.com/bzhanglab/proms/main/docs/svgs/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.075367949999992pt height=22.831056599999986pt/> medoids are selected as the final protein markers for training a classifier. The steps are depicted in the following diagram:

<center><img src="https://raw.githubusercontent.com/bzhanglab/proms/main/docs/proms_mo.png" alt="proms" height="800"/></center>


## Data preparation

### Training data

In order to train a model, we need the primary data set (e.g. proteomics data) and its corresponding phenotype data.  Primary data set should be stored in a tab separated values (`.tsv`) format where columns represent individual samples of the dataset, and rows represent features (e.g. protein expression) related to each sample. The following figure shows the contents of an example data file:

<img src="https://raw.githubusercontent.com/bzhanglab/proms/main/docs/data_format.png" alt="data_format" width="600"/>

We assume the data are appropriately normalized and transformed. By default, `proms` will remove features with missing values in any sample. To avoid such removal, you should impute those values accordingly. The [scikit-learn documentation](https://scikit-learn.org/stable/modules/preprocessing.html) has some information on how to use various preprocessing methods. 

Phenotype data is also stored in a `.tsv` file where rows represent samples and columns represent the values of each phenotype label. The data configuration file specifies which label will be used for constructing the model. The following figure shows the contents of an example phenotype data file:

<img src="https://raw.githubusercontent.com/bzhanglab/proms/main/docs/label.png" alt="label" width="200"/>

You may also provide auxiliary data sets (e.g. other omics data) with the same format as the primary data file. `proms_train` will also employ `ProMS_mo` to select features if more than one data sets are provided for the same set of samples.  

### Test data

Independent test data are optional. If provided, only the overlapping features between the primary data set and test data set will be used for marker selection. The data should be stored in a `.tsv` file with the same format as the primary training data file. Optionally, user can provide a phenotype label file for the samples in the test data. In this case, accuracy and AUROC will be calculated. 


## Configuration files

To use `proms` for marker selection and model construction , two configuration files are needed. A `run_config_file` describes the settings for the run and hyper-parameters. It is a json file with the following schema:


| Key         | Description                                                                                                      | Type   |
|-------------|------------------------------------------------------------------------------------------------------------------|--------|
| repeat      | Number of cross validation repeats                                                                               | number |
| k           | Number of selected markers                                                                                       | array  |
| classifiers | One model will be trained for each classifier with selected markers as features                                  | array  |
| percentile  | Percent of features to keep in the filtering step, the algorithm will determine the best "percentile" to be used | array  |
| n_jobs      | maximum number of concurrently running workers                                                                   | number |

Currently, the following classifiers are supported: 

* logistic regression (`logreg`),
* support vector machines (`svm`),
* eXtreme Gradient Boosting (`xgboost`),
* random forest (`rf`)

An example run configuration file is shown below:
```
{
  "repeat": 10,
  "k": [5,10,15],
  "classifiers": ["logreg", "svm"],
  "percentile": [5, 10, 15],
  "n_jobs": 4
}
```

`data_config_file` describes the input data. It is a json file with the following schema:

| Key                     | Description                                                       | Type   |
|-------------------------|-------------------------------------------------------------------|--------|
| name                    | name of the dataset                                               | string |
| data_root               | root directory of data folder relative to the config file         | string |
| train_dataset           | path of train dataset relative to data_root                       | string |
| test_dataset            | path of test dataset relative to data_root (optional)             | string |
| target_view             | name of data view where markers will be selected                  | string |
| target_label            | name of label to predict                                          | string |
| data                    | description of actual data                                        | object |
| data: train              | object describing train data                                      | object |
| data: train : label        | object describing class label information                         | object |
| data: train : label : file   | name of file with class label information                         | string |
| data: train : view         | a list of objects describing training omics data type             | array  |
| data: train : view[i] :type | name of omics data type i                                         | string |
| data: train : view[i] :file | name of file for omics data type i                                | string |
| data: test               | object describing test data (optional)                            | object |
| data: test : label         | pbject describing class label information (optional)              | object |
| data: test : label :file    | name of file with class label information (optional)              | string |
| data: test : view          | array of single object describing test omics data type (optional) | array  |
| data: test : view[0] :type  | name of target omics data type (optional, must match target_view) | string |
| data: test : view[0] :file  | name of file for target omics data type                           | string | 

An example data configuration file is show below:

```
{
  "name": "crc",
  "data_root": "crc",
  "train_dataset": "train",
  "test_dataset": "test",
  "target_view": "pro",
  "target_label": "msi",
  "data": {
    "train": {
      "label": {
        "file": "clinical_data.tsv"
      },
      "view": [
        {
          "type": "mrna",
          "file": "Colon_rna_fpkm.tsv"
        },
        {
          "type": "pro",
          "file": "Colon_pro_spc.tsv"
        }
      ]
    },
    "test": {
      "label": {
        "file": "clinical_data.tsv"
      },
      "view": [
        {
          "type": "pro",
          "file": "Colon_pro_spc_2.tsv"
        }
      ]
    }
  }
}
```
## How to run

### Training/validation/test

To use ProMS algorithms, run `proms_train`. The required arguments are `-f run_config_file` and `-d data_config_file`. `proms_train` will perform training and validation using the training data. After training/validation, a final full model will be trained with all train data with hyper-parameters that achieves the best average cross validation performance. If test data is provided in the data configuration file, `proms_train` will also apply the full model to the test data.

To see all available arguments for the command:

```console
> proms_train -h
usage: proms_train [-h] -f FILE -d FILE [-s SEED] [-o OUTPUT_ROOT]
                   [-r RUN_VERSION]

optional arguments:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  configuration file for the run
  -d FILE, --data FILE  configuration file for data set
  -s SEED, --seed SEED  random seed
  -o OUTPUT_ROOT, --output OUTPUT_ROOT
                        output directory
  -r RUN_VERSION, --run_version RUN_VERSION
                        name of the run, default to current date/time

```


## Output files
After the run, two `.tsv` files will be generated. One file with name 
ending with `_eval.tsv` has the following columns:

| Column name    | Description                                      |
|----------------|--------------------------------------------------|
| fs             | Feature selection method (`proms` or `proms_mo`) |
| type           | `so`(single omics) or `mo` (multi-omics)         |
| k              | number of selected markers                       |
| classifier     | one of the four supported classifier used        |
| repeat         | repeat index                                     |
| val_score      | score for the validation set (comma separated)                     |
| val_pred_label | predicted labels for the validation set  (comma separated)         |
| val_label      | true labels for the validation set (comma separated)               |
| val_acc        | accuracy for the validation set                  |
| val_auroc      | AUROC for the validation set                     |


An example file is shown below:

```
fs  type  k classifier  repeat  val_score val_pred_label  val_label val_acc val_auroc
proms so  15  logreg  0 0.1690,0.1718,0.1654,0.1675,0.1667,0.1686,0.1664,0.1710,0.1687,0.1722,0.1688,0.1698,0.1686,0.1691,0.1702,0.1718,0.1742,0.1700,0.1749,0.1694,0.1741,0.1666,0.1683,0.1692,0.1688,0.1721 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,0,0,0,1 0.8077  0.9524
proms_mo  mo  15  logreg  0 0.1692,0.1709,0.1645,0.1677,0.1688,0.1707,0.1638,0.1684,0.1675,0.1718,0.1697,0.1697,0.1696,0.1652,0.1679,0.1693,0.1730,0.1702,0.1735,0.1690,0.1749,0.1671,0.1685,0.1695,0.1705,0.1724 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,0,0,0,1 0.8077  0.8571
proms so  15  logreg  1 0.1673,0.1680,0.1680,0.1654,0.1693,0.1794,0.1737,0.1742,0.1663,0.1731,0.1719,0.1683,0.1726,0.1673,0.1704,0.1708,0.1693,0.1670,0.1726,0.1691,0.1697,0.1741,0.1685,0.1766,0.1713,0.1679 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 0,0,0,0,0,1,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0 0.8077  0.9619
proms_mo  mo  15  logreg  1 0.0027,0.0418,0.0060,0.0039,0.0065,0.9838,0.5603,0.8210,0.0034,0.3743,0.0092,0.0045,0.4444,0.0129,0.0512,0.0718,0.0212,0.0074,0.0772,0.0040,0.0376,0.0701,0.0218,0.9280,0.0531,0.0071 0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0 0,0,0,0,0,1,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0 0.8846  0.981
```

The name of the other `.tsv` file ends with `_full.tsv`.  This file contains the information about the final model and its performance on independent test data set (if available). It contains the following columns:

| Column name    | Description                                      |
|----------------|--------------------------------------------------|
| fs             | Feature selection method (`proms` or `proms_mo`) |
| type           | `so`(single omics) or `mo` (multi-omics)         |
| k              | number of selected markers                       |
| classifier     | classifier used                                  |
| features       | name of selected markers (comma separated)       |  
| membership     | a json string describe the membership of each cluster where keys are the final selected markers |
| avg_val_acc    | average validation accuracy for the selected markers/k/classifier combination |
| avg_val_auroc  | average validation AUROC for the selected markers/k/classifier combination |
| test_score     | score for the test data set (if test data set is provided)| 
| test_pred_label |predicted labels for the test data set (if test data set is provided)| 
| test_label     | true labels for the test data set  (if test data labels are also provided)| 
| test_accuracy  | accuracy for the test data set (if test data labels are also provided)| 

The final full model is generated as `full_model/full_model.pkl`. This can be used to make predictions on new data set.


## Prediction on new data set

To make predictions on new dataset with the trained full model, run: 

```console
proms_predict -m /path/to/saved/full_model.pkl -d predict_data_config
```

`predict_data_config` is a json file with the following schema:


| Key                     | Description                                                           | Type   |
|-------------------------|-------------------------------------------------------------------|--------|
| name                    | name of the dataset                                               | string |
| data_root               | root directory of data folder relative to the config file         | string |
| predict_dataset         | path of predict dataset relative to data_root                     | string |
| data                    | description of actual data                                        | object |
| data: predict            | object describing predict data                                    | object |
| data: predict : view       | array of single object describing preidction omics data type      | array  |
| data: test : view[0] : type  | name of target omics data type (must match training target_view)  | string |
| data: test : view[0] : file  | name of file for prediction omics data type                       | string |

An example predict data configuration file is shown below:
```
{
  "name": "crc",
  "data_root": "crc",
  "predict_dataset": "predict",
  "data": {
    "predict": {
      "view": [
        {
          "type": "pro",
          "file": "new_color_pro.tsv"
        }
      ]
    }
  }
}
```


## Example

To test the package, download a sample dataset from [here](https://zhanglab.s3-us-west-2.amazonaws.com/proms_test.tgz). 


```console
tar xvfz proms_test.tgz
cd proms_test
proms_train -f ./test/run_config.json -d ./data/crc.json 2>proms.log
```
Here we redirect all outputs other than stdout to a log file.

The cross validation results will be created as:
`[TIME_STAMP]` is the time stamp when the program started. By default, this 
is used to distinguish different runs.

`results/[TIME_STAMP]/crc_results_[TIME_STAMP]_eval.tsv`

The prediction results on test dataset with full model will be created as:

`results/[TIME_STAMP]/crc_results_[TIME_STAMP]_full.tsv`

The full model is saved as 

`results/[TIME_STAMP]/full_model/full_model.pkl`

To make prediction on another dataset with trained full model. (Here, we again make prediction on the test dataset.)


Note: replace [TIME_STAMP] with actual value.
```console
proms_predict -m results/[TIME_STAMP]/full_model/full_model.pkl -d data/crc_predict.json
```
