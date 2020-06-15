from lib.dataset import citation


def get_dataset(name,
                normalize_feature=True,
                normalize_adj=False,
                add_noise_to_graph=False):
    if name == 'citeseer':
        data = citation.load(
            'citeseer', normalize_feature,
            normalize_adj, add_noise_to_graph)
        out_features = 6
    elif name == 'cora':
        data = citation.load(
            'cora', normalize_feature,
            normalize_adj, add_noise_to_graph)
        out_features = 7
    elif name == "pubmed":
        data = citation.load(
            'pubmed', normalize_feature,
            normalize_adj, add_noise_to_graph)
        out_features = 3
    else:
        raise ValueError("no such dataset is defined: " + name)

    in_features = data.X.shape[1]

    return data, in_features, out_features
