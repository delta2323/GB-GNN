import numpy as np

from lib.dataset import kipf
import lib.dataset.normalization as N
from lib.dataset import wrapper


def load(dataset_name,
         normalize_feature=True,
         normalize_adj=False,
         add_noise_to_graph=False):
    """Loads citation dataset

    Args:
        dataset_name (str): choice of dataset
        gcn_normalize (Boolean): Set True if used for RSGCN
    """

    features, labels, adj, indices = kipf.load(
        dataset_name, add_noise_to_graph)

    if normalize_feature:
        features = N.row_normalize(features)
    if normalize_adj:
        adj = N.augmented_normalize(adj)

    features = features.toarray().astype(np.float32)
    adj = adj.toarray().astype(np.float32)

    return wrapper.DatasetWrapper(
        features, adj, labels, indices)
