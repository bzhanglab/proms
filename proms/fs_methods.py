import numpy as np
import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif
from .k_medoids import KMedoids
from .utils import sym_auc_score
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectPercentile
import warnings
from sklearn.feature_selection._univariate_selection import _BaseFilter
from sklearn.feature_selection._univariate_selection import _clean_nans
from sklearn.utils.validation import check_is_fitted


class SelectMinScore(_BaseFilter):
    """
    select features with minimum score threshold
    """

    def __init__(self, score_func=f_classif, min_score=0.5):
        super(SelectMinScore, self).__init__(score_func)
        self.min_score = min_score
        self.score_func = score_func

    def _check_params(self, X, y):
        if not (self.min_score == "all" or 0 <= self.min_score <= 1):
            raise ValueError('min_score should be >=0, <= 1; got {}.'
                             'Use min_score="all" to return all features.'
                             .format(self.min_score))

    def _get_support_mask(self):
        check_is_fitted(self, 'scores_')
        if self.min_score == 'all':
            return np.ones(self.scores_.shape, dtype=bool)

        scores = _clean_nans(self.scores_)
        mask = np.zeros(scores.shape, dtype=bool)
        mask = scores > self.min_score
        # Note: it is possible that mask contains all False.
        return mask


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, views, target_view, method, k=3,
                 weighted=False, percentile=1):
        self.views = views
        self.target_view = target_view
        self.method = method
        self.mode = self.get_filter_mode()
        self.k = k
        self.weighted = weighted
        self.percentile = percentile
        self.cluster_membership = None
        self.selected_features = None
        self.data = {}
        self.support = {}
    
    def get_filter_mode(self):
        if self.method.endswith('_mo'):
            return 'mo'
        return 'so'

    def feature_sel(self):
        fs_method = fs_methods[self.method]
        if self.method in ['proms', 'proms_mo']:
            # cluster_membership is a dictionary with selected
            # markers as keys
            ret = fs_method(self.data, self.target_view,
                            self.k, self.weighted)()
            selected_features, cluster_membership = ret
            return (selected_features, cluster_membership)
        else:
            raise ValueError('method {} is not supported'.format(self.method))

    def assemble_data(self, X, y=None):
        ''' X is a combined multi-view data frame
        '''
        for view in self.views:
            # find features in each view
            ptn = re.compile(r'^{}_'.format(view))
            cur_view_features = [i for i in self.all_features
                                 if ptn.match(i)]
            cur_X = X.loc[:, cur_view_features]
            self.data[view] = {}
            self.data[view]['X'] = cur_X
            self.data[view]['y'] = y

    def single_view_prefilter(self, X, y=None):
        view = self.views[0]
        if view != self.target_view:
            raise ValueError('target view name not matched')
        ptn = re.compile(r'^{}_'.format(view))
        cur_view_features = [i for i in self.all_features
                             if ptn.match(i)]
        cur_X = X.loc[:, cur_view_features]
        selector1 = SelectPercentile(sym_auc_score,
                                     percentile=self.percentile)
        selector1.fit(cur_X, y)
        support = selector1.get_support()
        self.support[view] = support
        self.data[view] = {}
        self.data[view]['X'] = cur_X.loc[:, support]
        self.data[view]['y'] = y

    def multi_view_prefilter(self, X, y=None):
        non_target_views = self.views.copy()
        non_target_views.remove(self.target_view)
        # target view
        # find features in each view
        ptn = re.compile(r'^{}_'.format(self.target_view))
        cur_view_features = [i for i in self.all_features
                             if ptn.match(i)]
        cur_X = X.loc[:, cur_view_features]
        selector1 = SelectPercentile(sym_auc_score,
                                     percentile=self.percentile)
        selector1.fit(cur_X, y)
        # find the score cutoff
        scores = selector1.scores_
        cutoff = np.sort(scores[selector1.get_support()])[0]
        support = selector1.get_support()
        self.support[self.target_view] = support
        self.data[self.target_view] = {}
        self.data[self.target_view]['X'] = cur_X.loc[:, support]
        self.data[self.target_view]['y'] = y

        for view in non_target_views:
            # find features in each view
            ptn = re.compile(r'^{}_'.format(view))
            cur_view_features = [i for i in self.all_features
                                 if ptn.match(i)]
            cur_X = X.loc[:, cur_view_features]
            selector1 = SelectMinScore(score_func=sym_auc_score,
                                       min_score=cutoff)
            selector1.fit(cur_X, y)
            support = selector1.get_support()
            self.support[view] = support
            if support.sum() > 0:
                self.data[view] = {}
                self.data[view]['X'] = cur_X.loc[:, support]
                self.data[view]['y'] = y

        if len(self.data) == 1:
            warnings.warn('non target views contributed zero feature.')

    def pre_filter(self, X, y=None):
        ''' X is a combined multi-view data frame
            prefiltering for each view
        '''
        if self.mode == 'so':
            self.single_view_prefilter(X, y)
        else:
            self.multi_view_prefilter(X, y)

    def fit(self, X, y=None):
        self.all_features = X.columns.tolist()
        self.pre_filter(X, y)
        self.results = self.feature_sel()
        self.selected_features = self.results[0]
        if len(self.results) == 2:
            self.cluster_membership = self.results[1]
        return self

    def get_feature_names(self):
        return self.selected_features

    def get_cluster_membership(self):
        return self.cluster_membership

    def transform(self, X, y=None):
        if self.selected_features is not None:
            return X.loc[:, self.selected_features]

class FeatureSelBase(object):
    """Base class for feature selection method"""
    def __init__(self, method_type, all_view_data, target_view_name, k):
        # type can be 'sv' (single view) or 'mv' (multi-view)
        self.method_type = method_type
        self.all_view_data = all_view_data
        self.target_view_name = target_view_name
        self.k = k

    def check_enough_feature(self, X, k):
        len_features = len(X.columns.values)
        warn_msg = 'not enough features in the target view'
        if len_features <= k:
            warnings.warn(warn_msg)

    def compute_feature_score(self, X, y, score_func=sym_auc_score):
        score_func_ret = score_func(X, y)
        if isinstance(score_func_ret, (list, tuple)):
            scores_, pvalues_ = score_func_ret
            pvalues_ = np.asarray(pvalues_)
        else:
            scores_ = score_func_ret
            pvalues_ = None
        # for now ignore pvalues
        return scores_


class ProMS(FeatureSelBase):
    """ProMS single view"""
    def __init__(self, all_view_data, target_view_name, k,
                 weighted=True):
        self.weighted = weighted
        super().__init__('sv', all_view_data, target_view_name, k)

    def __call__(self):
        # use default parameters
        km = KMedoids(n_clusters=self.k, init='k-medoids++', max_iter=300,
                      metric='correlation', random_state=0)
        # feature wise k-medoids clustering
        X = self.all_view_data[self.target_view_name]['X']
        y = self.all_view_data[self.target_view_name]['y']
        self.check_enough_feature(X, self.k)

        if self.weighted:
            feature_scores = self.compute_feature_score(X, y)

        all_target_feature_names = X.columns.values
        X = X.T
        if self.weighted:
            km.fit(X, sample_weight=feature_scores)
        else:
            km.fit(X)
        km.fit(X)

        # the cluster centers are real data points
        selected_target_features_idx = km.medoid_indices_
        # print out the class membership
        cluster_label = km.labels_
        cluster_membership = dict()
        selected_features = []
        for i, _ in enumerate(selected_target_features_idx):
            cur_idx = selected_target_features_idx[i]
            cur_label = cluster_label[cur_idx]
            cur_selected = all_target_feature_names[cur_idx]
            selected_features.append(cur_selected)
            cluster_members = []
            indices = [j for j, x in enumerate(cluster_label)
                       if x == cur_label]
            # remove self
            indices.remove(cur_idx)
            if len(indices) > 0:
                cluster_members = [all_target_feature_names[j] for j in
                                   indices]
            else:
                cluster_members = []
            cluster_membership[cur_selected] = cluster_members
        return (selected_features, cluster_membership)


class ProMS_mo(FeatureSelBase):
    """Multiomics ProMS"""
    def __init__(self, all_view_data, target_view_name, k,
                 weighted=True):
        self.weighted = weighted
        super().__init__('mv', all_view_data, target_view_name, k)

    def __call__(self):
        km = KMedoids(n_clusters=self.k, init='k-medoids++', max_iter=300,
                      metric='correlation', random_state=0)
        all_X = None
        all_features = pd.DataFrame(columns=['name', 'view'])
        if self.weighted:
            feature_scores = np.array([], dtype=np.float64)

        candidacy = np.array([], dtype=np.bool)

        for i in self.all_view_data:
            X = self.all_view_data[i]['X']
            y = self.all_view_data[i]['y']
            if i == self.target_view_name:
                self.check_enough_feature(X, self.k)

            all_X = pd.concat([all_X, X], axis=1)
            cur_features = pd.DataFrame(columns=['name', 'view'])
            cur_features['name'] = X.columns.values
            cur_features['view'] = i
            all_features = pd.concat([all_features, cur_features], axis=0)
            candidacy = np.concatenate((candidacy, np.repeat(i ==
                                       self.target_view_name,
                                       len(cur_features.index))))
            if self.weighted:
                cur_feature_scores = self.compute_feature_score(X, y)
                feature_scores = np.concatenate((feature_scores,
                                                cur_feature_scores))

        all_feature_names = all_X.columns.values
        # feature wise k-medoids clustering
        # all_X is now of shape:  n_features x n_sample
        all_X = all_X.T
        if self.weighted:
            km.fit(all_X, sample_weight=feature_scores, candidacy=candidacy)
        else:
            km.fit(all_X, candidacy=candidacy)

        # the cluster centers are real data points
        selected_target_features_idx = km.medoid_indices_
        # print out the class membership
        cluster_label = km.labels_
        cluster_membership = dict()
        selected_features = []
        for i, _ in enumerate(selected_target_features_idx):
            cur_idx = selected_target_features_idx[i]
            cur_label = cluster_label[cur_idx]
            cur_selected = all_feature_names[cur_idx]
            selected_features.append(cur_selected)
            cluster_members = []
            indices = [j for j, x in enumerate(cluster_label)
                       if x == cur_label]
            # remove self
            indices.remove(cur_idx)
            if len(indices) > 0:
                cluster_members = [all_feature_names[j] for j in
                                   indices]
            else:
                cluster_members = []
            cluster_membership[cur_selected] = cluster_members
        return (selected_features, cluster_membership)


fs_methods = {
    'proms': ProMS,
    'proms_mo': ProMS_mo
}
