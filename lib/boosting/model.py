from ignite import engine
import torch
from torch import nn

from lib.model.mlp import MLP
from lib import util


class Model(nn.Module):

    def __init__(self, in_features, n_units,
                 out_features, n_layers, args):
        super().__init__()
        self.mlp = MLP(
            [in_features] + [n_units] * n_layers + [out_features],
            args['dropout'], bias=True)
        self.args = args

    def __call__(self, X):
        return self.mlp(X)

    def fit(self, X, y, w):
        def loss_fn(z, y):
            loss = nn.CrossEntropyLoss(reduction='none')(z, y)
            return (loss * w).sum()

        optimizer = util.get_optimizer(
            self.args['optimizer'],
            self.mlp.parameters(),
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
