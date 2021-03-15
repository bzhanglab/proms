Data preparation
================
.. _training-data:

Training data 
-------------

In order to perform feature selection and train a model, we need the primary 
data set (e.g. proteomics data) and its corresponding label data. 
Primary data set should be stored in a tab separated values (.tsv) format
where columns represent individual samples of the dataset,
and rows represent features (e.g. protein expression) related to each sample.
The following figure shows the contents of an example data file:

.. image:: images/data_format.png
  :align: center
  :width: 500px


We assume that the data are appropriately normalized and transformed. 
By default, `proms` will remove features with missing values in any sample.
To avoid such removal, you should impute those values accordingly. 
The scikit-learn `documentation <https://scikit-learn.org/stable/modules/preprocessing.html>`_ 
has some information on how to use various preprocessing methods.


Label data is also stored in a .tsv file where rows represent samples 
and columns represent the values of each label. The :ref:`data-config-file` 
specifies the name of label that will be used for constructing the model. 
The following figure shows the contents of an example label data file:

.. image:: images/label.png
  :align: center
  :width: 150px

.. Note::
  For binary classification task, the valid label values are either 0 (negative class)
  or 1 (positive class). For surival analysis task, columns for  
  both event indicator and survival/censoring time should be provided.

You may also provide auxiliary data sets (e.g. other omics data) with the same 
format as the primary data file. `ProMS` will also employ ``ProMS_mo`` 
to select features if more than one data sets are provided for the same set of samples.   

Test data 
---------
Independent test data are optional. If provided, only the overlapping features 
between the primary data set and test data set will be used for marker selection.
The data should be stored in a .tsv file with the same format as the primary
training data file. Optionally, user can provide a label file for the samples 
in the test data. In this case, prediction performance will be evaluated based
on the true labels.