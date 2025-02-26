from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import InputSpec
import tensorflow.keras.backend as K
import numpy as np
from statsmodels.tsa import stattools
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


def eucl(x, y):
    """
    Euclidean distance between two multivariate time series given as arrays of shape (timesteps, dim)
    """
    d = np.sqrt(np.sum(np.square(x - y), axis=0))
    return np.sum(d)


def cid(x, y):
    """
    Complexity-Invariant Distance (CID) between two multivariate time series given as arrays of shape (timesteps, dim)
    Reference: Batista, Wang & Keogh (2011). A Complexity-Invariant Distance Measure for Time Series. https://doi.org/10.1137/1.9781611972818.60
    """
    assert(len(x.shape) == 2 and x.shape == y.shape)  # time series must have same length and dimensionality
    ce_x = np.sqrt(np.sum(np.square(np.diff(x, axis=0)), axis=0) + 1e-9)
    ce_y = np.sqrt(np.sum(np.square(np.diff(y, axis=0)), axis=0) + 1e-9)
    d = np.sqrt(np.sum(np.square(x - y), axis=0)) * np.divide(np.maximum(ce_x, ce_y), np.minimum(ce_x, ce_y))
    return np.sum(d)


def cor(x, y):
    """
    Correlation-based distance (COR) between two multivariate time series given as arrays of shape (timesteps, dim)
    """
    scaler = TimeSeriesScalerMeanVariance()
    x_norm = scaler.fit_transform(x)
    y_norm = scaler.fit_transform(y)
    pcc = np.mean(x_norm * y_norm)  # Pearson correlation coefficients
    d = np.sqrt(2.0 * (1.0 - pcc + 1e-9))  # correlation-based similarities
    return np.sum(d)


def acf(x, y):
    """
    Autocorrelation-based distance (ACF) between two multivariate time series given as arrays of shape (timesteps, dim)
    Computes a linearly weighted euclidean distance between the autocorrelation coefficients of the input time series.
    Reference: Galeano & Pena (2000). Multivariate Analysis in Vector Time Series.
    """
    assert (len(x.shape) == 2 and x.shape == y.shape)  # time series must have same length and dimensionality
    x_acf = np.apply_along_axis(lambda z: stattools.acf(z, nlags=z.shape[0]), 0, x)
    y_acf = np.apply_along_axis(lambda z: stattools.acf(z, nlags=z.shape[0]), 0, y)
    weights = np.linspace(1.0, 0.0, x.shape[0])
    d = np.sqrt(np.sum(np.expand_dims(weights, axis=1) * np.square(x_acf - y_acf), axis=0))
    return np.sum(d)

class TSClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, timesteps, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
        dist_metric: distance metric between sequences used in similarity kernel ('eucl', 'cir', 'cor' or 'acf').
    # Input shape
        3D tensor with shape: `(n_samples, timesteps, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, dist_metric='eucl', **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(TSClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.dist_metric = dist_metric
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=3)
        self.clusters = None
        self.built = False

    def build(self, input_shape):
        assert len(input_shape) == 3
        input_dim = input_shape[2]
        input_steps = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_steps, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_steps, input_dim), initializer='glorot_uniform', name='cluster_centers')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """
        Student t-distribution kernel, probability of assigning encoded sequence i to cluster k.
            q_{ik} = (1 + dist(z_i, m_k)^2)^{-1} / normalization.
        Arguments:
            inputs: encoded input sequences, shape=(n_samples, timesteps, n_features)
        Return:
            q: soft labels for each sample. shape=(n_samples, n_clusters)
        """
        if self.dist_metric == 'eucl':
            distance = K.sum(K.sqrt(K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2)), axis=-1)
        elif self.dist_metric == 'cid':
            ce_x = K.sqrt(K.sum(K.square(inputs[:, 1:, :] - inputs[:, :-1, :]), axis=1))  # shape (n_samples, n_features)
            ce_w = K.sqrt(K.sum(K.square(self.clusters[:, 1:, :] - self.clusters[:, :-1, :]), axis=1))  # shape (n_clusters, n_features)
            ce = K.maximum(K.expand_dims(ce_x, axis=1), ce_w) / K.minimum(K.expand_dims(ce_x, axis=1), ce_w)  # shape (n_samples, n_clusters, n_features)
            ed = K.sqrt(K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2))  # shape (n_samples, n_clusters, n_features)
            distance = K.sum(ed * ce, axis=-1)  # shape (n_samples, n_clusters)
        elif self.dist_metric == 'cor':
            inputs_norm = (inputs - K.expand_dims(K.mean(inputs, axis=1), axis=1)) / K.expand_dims(K.std(inputs, axis=1), axis=1)  # shape (n_samples, timesteps, n_features)
            clusters_norm = (self.clusters - K.expand_dims(K.mean(self.clusters, axis=1), axis=1)) / K.expand_dims(K.std(self.clusters, axis=1), axis=1)  # shape (n_clusters, timesteps, n_features)
            pcc = K.mean(K.expand_dims(inputs_norm, axis=1) * clusters_norm, axis=2)  # Pearson correlation coefficients
            distance = K.sum(K.sqrt(2.0 * (1.0 - pcc)), axis=-1)  # correlation-based similarities, shape (n_samples, n_clusters)
        elif self.dist_metric == 'acf':
            raise NotImplementedError
        else:
            raise ValueError('Available distances are eucl, cid, cor and acf!')
        q = 1.0 / (1.0 + K.square(distance) / self.alpha)
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 3
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters, 'dist_metric': self.dist_metric}
        base_config = super(TSClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))