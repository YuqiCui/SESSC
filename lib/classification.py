import numpy as np
from lib.cluster import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.linear_model import RidgeClassifier, MultiTaskLasso


fuzzy_cluster_dict = {
    'fcm': FuzzyCMeans,
    'essc': ESSC,
    'sessc': SESSC,
}
fuzzy_cls_dict = {
    'ridge': RidgeClassifier,
}


def _fuzzy_cluster_(n_cluster, name='fcm'):
    return fuzzy_cluster_dict[name.lower()](n_cluster=n_cluster)


def _fuzzy_cls_(name='ridge'):
    return fuzzy_cls_dict[name.lower()]()


def x2xp(cluster_model, X, order=1):
    if order == 0:
        return cluster_model.predict(X)
    else:
        N = X.shape[0]
        mem = np.expand_dims(cluster_model.predict(X), axis=1)
        X = np.expand_dims(np.concatenate((X, np.ones([N, 1])), axis=1), axis=2)
        X = np.repeat(X, repeats=cluster_model.n_cluster, axis=2)
        xp = X * mem
        xp = xp.reshape([N, -1])
        return xp


class TSK_FS(BaseEstimator, ClassifierMixin):
    def __init__(self, n_cluster=5, order=0, cluster='fcm', classifier='ridge', norm_sparse_thres=0., **kwargs):
        self.cluster = cluster
        self.classifier = classifier
        self.order = order

        self.cluster_app = _fuzzy_cluster_(n_cluster, name=cluster)
        self.classifier_app = _fuzzy_cls_(classifier)

        self.norm_sparse_thres = norm_sparse_thres
        self.set_params(**kwargs)

    def get_params(self, deep=True):
        self_param = {
            'cluster': self.cluster,
            'classifier': self.classifier,
            'order': self.order,
            'norm_sparse_thres': self.norm_sparse_thres
        }
        cluster_param = self.cluster_app.get_params()
        cls_param = self.classifier_app.get_params()

        keys = list(cluster_param.keys())
        for k in keys:
            cluster_param['_cluster_'+k] = cluster_param.pop(k)

        keys = list(cls_param.keys())
        for k in keys:
            cls_param['_cls_' + k] = cls_param.pop(k)

        self_param.update(cluster_param)
        self_param.update(cls_param)
        return self_param

    def set_params(self, **params):
        new_cluster = params.pop('cluster', None)
        if new_cluster is not None and new_cluster != self.cluster:
            self.cluster_app = _fuzzy_cluster_(
            self.cluster_app.n_cluster, new_cluster
        )

        new_cls = params.pop('classifier', None)
        if new_cls is not None and new_cls != self.classifier:
            self.classifier_app = _fuzzy_cls_(new_cls)

        for k, v in params.items():
            if '_cluster_' in k:
                setattr(self.cluster_app, k.replace('_cluster_', ''), v)
            elif '_cls_' in k:
                setattr(self.classifier_app, k.replace('_cls_', ''), v)
            else:
                setattr(self, k, v)
        return self

    def fit(self, X, y=None):
        X = np.array(X, dtype=np.float64)
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        self.cluster_app.fit(X, y)
        xp = x2xp(self.cluster_app, X, order=self.order)
        self.classifier_app.fit(xp, y)
        self.fitted = True
        return self

    def predict(self, X):
        check_is_fitted(self, ['fitted'])
        X = check_array(X)

        xp = x2xp(self.cluster_app, X, order=self.order)
        pred = self.classifier_app.predict(xp)
        if pred.ndim > 1:
            return np.argmax(pred, axis=1)

        return pred


class SESSC_Raw_Pred(BaseEstimator, ClassifierMixin):
    def __init__(self, n_cluster, scale=1., m='auto', eta=0.1, gamma=0.1, beta=0.1,
                 error=1e-5, tol_iter=200, verbose=0, init='kmean'):
        self.n_cluster = n_cluster
        self.scale = scale
        self.m = m
        self.eta = eta
        self.gamma = gamma
        self.beta = beta
        self.error = error
        self.tol_iter = tol_iter
        self.verbose = verbose
        self.init = init

    def get_params(self, deep=True):
        return {
            'n_cluster': self.n_cluster,
            'scale': self.scale,
            'm': self.m,
            'eta': self.eta,
            'gamma': self.gamma,
            'beta': self.beta,
            'error': self.error,
            'tol_iter': self.tol_iter,
            'verbose': self.verbose,
            'init': self.init
        }

    def set_params(self, **params):
        for p, v in params.items():
            setattr(self, p, v)
        return self

    def fit(self, X, y=None):
        X = np.array(X, dtype=np.float64)
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        self.cluster = SESSC(**self.get_params())
        self.cluster.fit(X, y)
        self.fitted = True
        return self

    def predict(self, X):
        return self.cluster.predict_label(X)


class SESSCL2_Raw_Pred(BaseEstimator, ClassifierMixin):
    def __init__(self, n_cluster, scale=1., m='auto', eta=0.1, gamma=0.1, beta=0.1,
                 error=1e-5, tol_iter=200, verbose=0, init='kmean'):
        self.n_cluster = n_cluster
        self.scale = scale
        self.m = m
        self.eta = eta
        self.gamma = gamma
        self.beta = beta
        self.error = error
        self.tol_iter = tol_iter
        self.verbose = verbose
        self.init = init

    def get_params(self, deep=True):
        return {
            'n_cluster': self.n_cluster,
            'scale': self.scale,
            'm': self.m,
            'eta': self.eta,
            'gamma': self.gamma,
            'beta': self.beta,
            'error': self.error,
            'tol_iter': self.tol_iter,
            'verbose': self.verbose,
            'init': self.init
        }

    def set_params(self, **params):
        for p, v in params.items():
            setattr(self, p, v)
        return self

    def fit(self, X, y=None):
        X = np.array(X, dtype=np.float64)
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        self.cluster = ESSC.SESSC_L2Label(**self.get_params())
        self.cluster.fit(X, y)
        self.fitted = True
        return self

    def predict(self, X):
        return self.cluster.predict_label(X)
