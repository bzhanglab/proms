Output files
============

After training, two .tsv files will be generated. One file with name ending
with _eval.tsv has the following columns (for classification task):

.. list-table::
   :widths: 25 50
   :header-rows: 1

   * - Column name
     - Description
   * - fs
     - Feature selection method (``proms`` or ``proms_mo`` or ``pca_ex``)
   * - type
     - multiview (``mo``) or single view (``so``)
   * - k
     - number of selected markers
   * - estimator
     - one of the supported estimators used 
   * - repeat
     - repeat index
   * - val_score
     - score for the validation set (comma separated)
   * - val_pred_label
     - predicted labels for the validation set (comma separated)
   * - val_label
     - true labels for the validation set (comma separated)
   * - val_acc
     - accuracy for the validation set
   * - val_auroc
     - AUROC for the validation set
    
The name of the other .tsv file ends with _full.tsv.
This file contains the information about the final model and its performance on
independent test data set (if available). It contains the following columns (for 
classification task):

.. list-table::
   :widths: 25 50
   :header-rows: 1

   * - Column name
     - Description
   * - fs
     - Feature selection method (``proms`` or ``proms_mo`` or ``pca_ex``)
   * - type
     - multiview (``mo``) or single view (``so``)
   * - k
     - number of selected markers
   * - estimator
     - the estimator used 
   * - features
     - name of selected markers (comma separated)
   * - membership
     - a json string describe the membership of each cluster where keys are the 
       final selected markers
   * - mean_val_acc
     - mean validation accuracy for the selected markers/k/estimator combination 
       among all evalutaion repeats
   * - mean_val_auroc
     - mean validation auroc for the selected markers/k/estimator combination 
       among all evalutaion repeats
   * - test_score
     - score for the test data set (if test data set is provided)
   * - test_pred_label
     - predicted labels for the test data set (if test data set is provided)
   * - test_label
     - true labels for the test data set (if test data labels are also provided)
   * - test_accuracy
     - accuracy for the test data set (if test data labels are also provided)
   * - test_auroc
     - auroc for the test data set (if test data labels are also provided)

The final full model is generated as `full_model/full_model.pkl`. This can be 
used for making predictions on new data set.