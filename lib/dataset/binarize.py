
from lib.dataset import wrapper


def binarize(dataset):
    data, in_features, out_features = dataset
    data = _binarize(data[0])
    data = wrapper.DatasetWrapper(data[0], data[1], data[2], data[3])
    out_features = 2
    return data, in_features, out_features


def _binarize(data):
    X, A, y, indices = data
    n_label = y.max() + 1
    pos_idx = y < n_label / 2
    neg_idx = y >= n_label / 2
    y[pos_idx] = 1
    y[neg_idx] = 0
    return X, A, y, indices
