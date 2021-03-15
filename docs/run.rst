How to run
==========

To use ProMS for selecting features and training a full model, run ``proms_train``. 
The required arguments are ``-f run_config_file`` and ``-d data_config_file``. 
``proms_train`` will perform training and validation using the training data.
After training/validation, a final full model will be trained with all train 
data with hyper-parameters that achieves the best average cross validation performance.
If test data is provided in the data configuration file, ``proms_train`` will
also apply the full model to the test data.

To see all available arguments for the command:

.. code-block:: none

    $ proms_train -h
    usage: proms_train [-h] -f FILE -d FILE [-s SEED] [-o OUTPUT_ROOT]
                    [-r RUN_VERSION] [-p]

    optional arguments:
    -h, --help            show this help message and exit
    -f FILE, --file FILE  configuration file for the run
    -d FILE, --data FILE  configuration file for data set
    -s SEED, --seed SEED  random seed
    -o OUTPUT_ROOT, --output OUTPUT_ROOT
                            output directory
    -r RUN_VERSION, --run_version RUN_VERSION
                            name of the run, default to current date/time
    -p, --include_pca     include supervised PCA method in the results