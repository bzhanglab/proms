Configuration files
===================
To use `proms` for marker selection and model construction, 
two configuration files are needed.

Run configuration file
----------------------

A run configuration file describes the settings for the run and hyperparameters.
It is a yaml file with the following schema:

.. list-table::
   :widths: 25 50
   :header-rows: 1

   * - Key
     - Description
   * - repeat
     - Number of cross validation repeats
   * - k
     - Number of selected markers
   * - estimators
     - One model will be trained for each estimator with selected markers as features
   * - percentile
     - Percent of features to keep in the filtering step, the algorithm will determine the best "percentile" to be used
   * - n_jobs
     - Maximum number of concurrently running workers

Currently, the following estimators are supported (name in the parentheses should be 
used in the configuration file)

For classification task:

- logistic regression (``lr``),
- support vector classifier (``svm``),
- eXtreme Gradient Boosting (``xgboost``),
- random forest (``rf``)

For regression task:

- ridge regression (``ridge``)
- support vector regressor (``svm``),
- eXtreme Gradient Boosting (``xgboost``),
- random forest (``rf``)

For survival analysi task:

- cox proportional hazards model (``coxph``)

An example run configuration file is shown below:

.. code-block:: yaml

    ---
    repeat: 2
    k:
    - 5
    - 10
    - 15
    estimators:
    - lr
    - rf 
    - svm 
    percentile:
    - 5
    - 10
    - 15
    n_jobs: 20




.. _data-config-file:

Data configuration file
-----------------------
A data configuration file describes the input data. It is a yaml file with the following schema:

.. list-table::
   :widths: 25 50
   :header-rows: 1

   * - Key
     - Description
   * - project_name
     - project short name
   * - data_directory
     - path to the data directory. It can be an absolute path or a directory name relative to the data configuration file.
   * - train_data_directory
     - path to the train data directory. It is relative to `data_directory`.
   * - test_data_directory
     - (**optional**) path to the independent test data directory. It is relative to `data_directory`.
   * - target_view
     - from which view should the markers be selected (see available views in the data/train/view section below)
   * - target_label
     - column name of attribute to be predicted 
   * - data
     - information about train and (optional) test data set
   * - data/train
     - information about train data set
   * - data/train/label/file
     - name of the file containing train labels
   * - data/train/label/view
     - a list of training data views. Each view consists of two items: *type* and *file* name
   * - data/test
     - (optional) information about test data set
   * - data/test/label/file
     - (optional) name of the file containing test labels
   * - data/test/label/view
     - (optional) a list of test data views. Each view consists of two items: *type* and *file* name


A sample data configuration file (`crc.yml`) is shown below:

.. code-block:: yaml

  ---
  project_name: crc
  data_directory: crc_data
  train_data_directory: train_data
  test_data_directory: test_data
  target_view: pro
  target_label: msi
  data:
    train:
      label:
        file: clinical_data_train.tsv
      view:
      - type: mrna
        file: Colon_rna_fpkm.tsv
      - type: pro
        file: Colon_pro_spc.tsv
    test:
      label:
        file: clinical_data_test.tsv
      view:
      - type: pro
        file: Colon_pro_spc_2.tsv

The corresponding directory structure is:

.. code-block:: none

   .
   ├── crc_data
   │   ├── test_data
   │   │   ├── Colon_pro_spc_2.tsv
   │   │   └── clinical_data_train.tsv
   │   └── train_data
   │       ├── Colon_pro_spc.tsv
   │       ├── Colon_rna_fpkm.tsv
   │       └── clinical_data_test.tsv
   └── crc.yml

