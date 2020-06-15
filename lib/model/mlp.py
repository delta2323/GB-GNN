import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, n_features,
                 dropout, bias):
        assert len(n_features) >= 2
        super(MLP, self).__init__()
        self.linears = nn.ModuleList(
            [nn.Linear(n_features[i],
                       n_features[i+1],
                       bias=bias)
             for i in range(len(n_features) - 1)])
        self.dropout = dropout

    def forward(self, x):
        h = x
        for l in self.linears[:-1]:
            h = F.relu(l(h))
            if self.dropout:
                h = F.dropout(h)
        h = self.linears[-1](h)
        return h
