from sklearn.metrics import roc_auc_score
from sklearn.utils import check_X_y
import numpy as np
# import pandas as pd
import multiprocessing
from sksurv.metrics import concordance_index_censored
# from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr


# class StandardScalerDf(StandardScaler):
#     """DataFrame Wrapper around StandardScaler"""
#     def __init__(self, copy=True, with_mean=True, with_std=True):
#         super().__init__(copy=copy, with_mean=with_mean, with_std=with_std)

#     def transform(self, X, copy=None):
#         z = super().transform(X.values)
#         return pd.DataFrame(z, index=X.index, columns=X.columns)


def _auc_score(y_pred, y_true):
    return roc_auc_score(y_true, y_pred)


# https://stackoverflow.com/a/45555516/410069
def unpacking_apply_along_axis(all_args):
    """
    Like numpy.apply_along_axis(), but with arguments in a tuple
    instead.

    This function is useful with multiprocessing.Pool().map(): (1)
    map() only handles functions that take a single argument, and (2)
    this function can generally be imported from a module, as required
    by map().
    """
    (func1d, axis, arr, args, kwargs) = all_args
    return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)


def parallel_apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
    Like numpy.apply_along_axis(), but takes advantage of multiple
    cores.
    """
    # Effective axis where apply_along_axis() will be applied by each
    # worker (any non-zero axis number would work, so as to allow the use
    # of `np.array_split()`, which is only done on axis 0):
    effective_axis = 1 if axis == 0 else axis
    if effective_axis != axis:
        arr = arr.swapaxes(axis, effective_axis)

    # Chunks for the mapping (only a few chunks):
    chunks = [(func1d, effective_axis, sub_arr, args, kwargs)
              for sub_arr in np.array_split(arr, multiprocessing.cpu_count())]

    pool = multiprocessing.Pool()
    individual_results = pool.map(unpacking_apply_along_axis, chunks)
    # Freeing the workers:
    pool.close()
    pool.join()

    return np.concatenate(individual_results)


def sym_auc_score(X, y):
    """Compute the symmetric auroc score for the provided sample.
    symmetric auroc score is defined as 2*abs(auroc-0.5)

    Parameters
    ----------
    X : {array-like, sparse matrix} shape = [n_samples, n_features]
        The set of regressors that will be tested sequentially.

    y : array of shape(n_samples)
        The data matrix.

    Returns
    -------
    F : array, shape = [n_features,]
        The set of auroc scores.
    """
    X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'])
    scores = np.apply_along_axis(_auc_score, 0, X, y)
    return np.abs(scores - 0.5) * 2.0


def _c_index_score(X, y):
    c_index = concordance_index_censored(y[y.dtype.names[0]],
                                         y[y.dtype.names[1]],
                                         X)
    return c_index[0]


def sym_c_index_score(X, y):
    """
    symmetric concordance index for right-censored data
    """
    scores = np.apply_along_axis(_c_index_score, 0, X, y)
    return np.abs(scores - 0.5) * 2.0


def _cor_coeff(X, y):
    """
    pearson correlation coefficient
    """
    corr, _ = pearsonr(X, y)
    return corr


def abs_cor(X, y):
    """
    symmetric concordance index for right-censored data
    """
    scores = np.apply_along_axis(_cor_coeff, 0, X, y)
    return np.abs(scores)
