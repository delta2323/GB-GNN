from scipy.spatial.distance import cosine
import torch
from torch import nn
from torch import optim


def get_device(device):
    if device >= 0:
        err_msg = (f'GPU={device} is specified. '
                   'Bug the GPU environment is not '
                   'correctly set up.')
        assert torch.cuda.is_available(), err_msg
        device = f'cuda:{device}'
    else:
        device = 'cpu'
    return device


def compute_loss(pred, y, indices):
    train_idx, val_idx, test_idx = indices
    loss_train = _compute_loss(pred[train_idx], y[train_idx])
    loss_val = _compute_loss(pred[val_idx], y[val_idx])
    loss_test = _compute_loss(pred[test_idx], y[test_idx])
    return loss_train, loss_val, loss_test


def _compute_loss(pred, y):
    pred = pred.to(y.device)
    return nn.NLLLoss()(pred, y).to('cpu')


def compute_loss_and_gradient(pred, y, indices):
    train_idx, val_idx, test_idx = indices
    loss_train, grad_train = _compute_loss_and_gradient(
        pred[train_idx], y[train_idx])
    loss_val, grad_val = _compute_loss_and_gradient(pred[val_idx], y[val_idx])
    loss_test, grad_test = _compute_loss_and_gradient(
        pred[test_idx], y[test_idx])
    return (loss_train, loss_val, loss_test), (grad_train, grad_val, grad_test)


def _compute_loss_and_gradient(pred, y):
    pred = pred.to(y.device)
    pred.requires_grad = True
    loss = nn.NLLLoss()(pred, y)
    grad = torch.autograd.grad([loss], [pred])[0].to('cpu')
    return loss, grad


def compute_cosine(x, y):
    # scipy.spatial.distance.cosine computes 1 - cos
    return 1 - cosine(x, y)


def get_optimizer(optimizer_name, parameters, lr, momentum, weight_decay):
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(
            parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            parameters,
            lr=lr,
            weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum)
    else:
        raise ValueError(f'Invalid optimizer: {optimizer}')
    return optimizer
