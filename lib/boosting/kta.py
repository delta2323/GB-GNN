from ignite import engine
import torch
from torch import nn
from torch import optim

from lib import util


def compute_A_powers(A, n_iter=3):
    ret = [
        torch.eye(len(A))[None].to(A.device).detach(),
        A[None].detach()
    ]
    A_now = A
    for _ in range(n_iter):
        A_now = A_now @ A_now
        ret.append(A_now[None].detach())
    return torch.cat(ret, 0)


def compute_ky(y):
    return (y[None] == y[:, None]).float()


class KTA(nn.Module):
    def __init__(self, As, ky, train_idx, args):
        self.As = As
        self.ky = ky
        self.train_idx = train_idx
        self.args = args
        super().__init__()
        self.weight = self._weight_init(
            args['weight_init_method'], len(As))

    def _weight_init(self, method, K):
        if method == 'all_one':
            return nn.Parameter(torch.ones(K))
        elif method == 'sum_to_one':
            return nn.Parameter(torch.ones(K) / K)
        else:
            raise ValueError(
                f'Invalid initialization method={method}')

    def __call__(self, X):
        weight = self.weight.to(X.device)
        X = X[0]
        X = self.As @ X
        X = torch.tensordot(weight, X, 1)
        return X

    def fit(self, X):
        def loss_fn(z, ky):
            kx = z @ z.T
            kx = kx[self.train_idx][:, self.train_idx]
            ky = ky[0][self.train_idx][:, self.train_idx]
            return -torch.nn.CosineSimilarity(0)(kx.flatten(), ky.flatten())

        optimizer = util.get_optimizer(
            self.args['optimizer'],
            self.parameters(),
            self.args['lr'],
            0., 0.)
        dataset = torch.utils.data.TensorDataset(X[None], self.ky[None])
        loader = torch.utils.data.DataLoader(dataset, 1, shuffle=False)
        device = X.device
        trainer = engine.create_supervised_trainer(
            self, optimizer, loss_fn, device=device)
        trainer.run(loader, max_epochs=self.args['n_epochs'])
        print('KTA weight', self.weight.detach().to('cpu').numpy())

    def transform(self, X):
        return self(X[None]).detach()


def update_X(X, As, ky, indices, args):
    model = KTA(As, ky, indices[0], args)
    model.fit(X)
    return model.transform(X), model
