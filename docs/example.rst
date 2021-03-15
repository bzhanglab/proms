Example
=======

To test the package, download a sample dataset from `here <https://zhanglab.s3-us-west-2.amazonaws.com/proms_test.tgz>`_.

.. code-block:: none

    tar xvfz proms_test.tgz
    cd proms_test
    proms_train -f ./test/run_config.yml -d ./data/crc.yml

The cross validation results will be created as: ([TIME_STAMP] is the time stamp 
when the program started. By default, this is used to distinguish different runs.)

.. code-block:: none 

    results/[TIME_STAMP]/crc_results_[TIME_STAMP]_eval.tsv

The prediction results on test dataset with full model will be created as:

.. code-block:: none 

    results/[TIME_STAMP]/crc_results_[TIME_STAMP]_full.tsv

The full model is saved as:

.. code-block:: none 

    results/[TIME_STAMP]/full_model/full_model.pkl

To make prediction on another dataset with trained full model. (Here, we again make prediction on the test dataset.)

.. code-block:: none 

    proms_predict -m results/[TIME_STAMP]/full_model/full_model.pkl -d data/crc_data/test_data/Colon_pro_spc_2.tsv