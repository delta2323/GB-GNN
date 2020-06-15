from ignite import engine
import numpy as np
import torch
from torch import nn

from lib.metric import metric
from lib import util


class Model(nn.Module):
    def __init__(self, weak_learners, weights, updaters, args, indices):
        self.args = args
        self.indices = indices
        super().__init__()
        self.weak_learners = nn.ModuleList(weak_learners)
        self.weights = nn.Parameter(torch.Tensor(weights))
        self.updaters = nn.ModuleList(updaters)
        self.to(args['device'])

    def __call__(self, X):
        y = None
        for i, (l, k) in enumerate(zip(self.weak_learners, self.updaters)):
            z = l(X)
            y_diff = torch.softmax(z, axis=1) * self.weights[i]
            y = y_diff if y is None else y + y_diff
            X = k(X[None])
        return y

    def fit(self, X, y):
        train_idx = self.indices[0]

        def loss_fn(z, y):
            return nn.CrossEntropyLoss()(z[train_idx], y[train_idx])

        optimizer = util.get_optimizer(
            self.args['optimizer'],
            self.parameters(),
            self.args['lr'],
            self.args['momentum'],
            self.args['weight_decay'])
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, len(X), shuffle=False)
        device = X.device
        trainer = engine.create_supervised_trainer(
            self, optimizer, loss_fn, device=device)
        trainer.run(loader, max_epochs=self.args['n_iters'])

    def predict(self, X):
        z = torch.softmax(self(X), dim=1)
        z.to('cpu')
        return z


def fine_tune(models, dataset, args):
    data, _, _ = dataset
    device = args['device']
    data.to(device)
    X, _, y, indices = data[0]

    weak_learners, weights, updaters = models
    model = Model(weak_learners, weights, updaters, args, indices)
    model.fit(X, y)
    pred = model.predict(X)
    loss = util.compute_loss(pred, y, indices)
    acc = metric.accuracy_all(pred, y, indices)

    ret = {
        'acc': {
            'train': acc[0],
            'val': acc[1],
            'test': acc[2]
        },
        'cosine': np.nan,  # placeholder
        'acc_all': {
            'train': np.nan,
            'val': np.nan,
            'test': np.nan
        },
        'loss': {
            'train': float(loss[0]),
            'val': float(loss[1]),
            'test': float(loss[2])
        }
    }
    return ret
