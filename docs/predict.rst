Prediction on new data set
==========================

To make predictions on new dataset with the full model, run:

.. code-block:: none

    proms_predict -m /path/to/saved/full_model.pkl -d /path/to/data_file

The data file should have the same format as :ref:`training-data`. The file should
contain measurments for all markers selected by the full model.

An output file (default name: `prediction_output.txt`) will be 
generated after the prediction.  For classification task, it contains 
3 columns: `sample`, `probability` (of being class 1), (predicted) `label`.