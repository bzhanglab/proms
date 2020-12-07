# ProMS: protein marker selection using proteomics or multiomics data


## Installation

To install `proms`, run:

```console
> pip install proms
```

or

```console
> pip install git+https://github.com/bzhanglab/proms
```


## Introduction

We provide two methods for selecting protein markers. 

### ProMS: Protein marker selection with proteomics data alone

The algorithm `ProMS` (Protein Marker Selection) works as follows. 
As a first step to remove uninformative features, `ProMS` examines each feature 
individually to determine the strength of the relationship between the feature 
and the target variable. A symmetric auroc score $AUC_{sym}$
is defined to evaluate such strength: $AUC_{sym} = 2 \times |AUC - 0.5|$ 


`ProMS` only keeps the features with the top $\alpha\%$ highest $AUC_{sym}$ 
scores. Here $\alpha$ is a hyperparameter that needs to be tuned jointly 
with other hyperparameters of the final classifier. After the filtering step, data 
matrix $\mathbf{D}$ is reduced to $\mathbf{D'}$ of size $n\times p'$ where  $p' \ll p$.
To further reduce the redundancy among the remaining features,
`ProMS` groups $p'$ features into $k$ clusters with weighted k-medoids clustering in sample space. 
The $k$ medoids from each cluster are selected as markers.
The whole process is illustrated in the following diagram:

<center><img src="https://github.com/bzhanglab/proms/blob/main/docs/proms.png" alt="proms" height="800"/></center>


### ProMS_mo: Protein marker selection with multi-omics data
We have $H$ data sources, $\mathbf{D}_1,...,\mathbf{D}_H$, representing $H$ 
different types of omics measurements that jointly depicts the same set of
samples $s_1,...,s_n$. $\mathbf{D}_i (i=1...H)$ is a matrix of size  $n\times p_i$ 
where rows correspond to samples and columns correspond to features in $i$th
data source. Without the loss of generality, we use $\mathbf{D}_1$ 
to represent the proteomics data from which we seek to select a set of informative 
markers that can be used to predict the target labels. Similar to `ProMS`, the first 
step of `ProMS_mo` involves filtering out insignificant features from each data source 
separately. Again we use $AUC_{sym}$. `ProMS_mo` first applies the univariate filtering to target data source $\mathbf{D}_1$ 
and keeps only the top $\alpha\%$ 
 features with the highest scores. We denote the minimal score among these remaining 
 features as $\theta$. For other data source, `ProMS_mo` only keeps those features 
 with score larger than $\theta$. Filtered data matrices are combined into a new 
 matrix $\mathbf{D}'$ of size $n\times p'$, where $p' = \sum_{i=1}^{H}p'_{i}$
 and $p'_i$ is the number of features in the filtered data source $i$. Finally, 
 weighted k-medoids clustering is performed to partition the $p'$ features into $k$
 clusters in sample spaces. To guarantee that only protein markers are selected as 
 medoids, `ProMS_mo` first initializes the $k$
 medoids to protein markers. During the iterative steps of optimization, 
 a medoid can only be replaced by another protein marker if such exchange improves the 
 objective function. After the iterative process converges, $k$
 medoids are selected as the final protein markers for training a classifier.
The steps are depicted in the following diagram:

<center><img src="https://github.com/bzhanglab/proms/blob/main/docs/proms_mo.png" alt="proms" height="800"/></center>

## How to use the package

To evaluate ProMS algorithms, use `proms_train`. The required arguments are `-f run_config_file` 
and `-d data_config_file`. After cross validation, a final full model will be trained with 
all train data with hyperparameters that achieves the best average cross validation performance.

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

`run_config_file` is a json file with the following schema:


| Key         | Description                                                                                                      | Type   |
|-------------|------------------------------------------------------------------------------------------------------------------|--------|
| repeat      | Number of cross validation repeats                                                                               | number |
| k           | Number of selected markers                                                                                       | array  |
| classifiers | One model will be trained for each classifier with selected markers as features                                  | array  |
| percentile  | Percent of features to keep in the filtering step, the algorithm will determine the best "percentile" to be used | array  |
| n_jobs      | maximum number of concurrently running workers                                                                   | number |

<br/><br/>

`data_config_file` is a json file with the following schema:

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

<br/><br/>

To make predictions on new dataset with the full model, run 

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

<br/>

### Example: 

To test the package, download the dataset from [here](https://zhanglab.s3-us-west-2.amazonaws.com/proms_test.tgz). 


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

To make prediction on another dataset with trained full model. (Here, we again
make prediction on the test dataset.)


Note: replace [TIME_STAMP] with actual value.
```console
proms_predict -m results/[TIME_STAMP]/full_model/full_model.pkl -d data/crc_predict.json
```

 

