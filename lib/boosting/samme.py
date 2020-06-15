import numpy as np
import torch
from torch import nn

from lib.boosting import model as model_
from lib.boosting import kta
from lib.metric import metric
from lib import util


def _compute_alpha(pred, y, w, K, clip_eps):
    def _clip(eps):
        eps = max(eps, clip_eps)
        eps = min(1 - clip_eps, eps)
        return eps

    incorrect_ = metric.incorrect(pred, y)
    # Note that since w is already normalized (i.e., sum to 1),
    # we do not need the division by w.sum()
    eps = float(w[incorrect_].sum())
    eps = _clip(eps)
    alpha = np.log((1 - eps) / eps) + np.log(K - 1)
    return alpha


def _update_weight(w, alpha, pred, y):
    incorrect_ = metric.incorrect(pred, y)
    mult = torch.ones(len(pred))
    mult[incorrect_] = np.exp(alpha)
    w_unnorm = w * mult
    return w_unnorm / w_unnorm.sum()


def _update_predict(pred, alpha, pred_one):
    ret = pred.clone()
    pred_label = pred_one.argmax(dim=1)
    ret[range(len(pred)), pred_label] += alpha
    diff = ret - pred
    return ret, diff


def _get_updater(train_args, X_init, A, y, indices):
    aggregation_model = train_args['aggregation_model']
    if aggregation_model == 'no_aggregation':
        class _Updater(nn.Module):
            def __call__(self, X):
                return X[0]

        def _updater(X):
            return X, _Updater()
        return _updater
    elif aggregation_model == 'ii':
        ii_args = train_args['ii_args']
        ratio = ii_args['aggregation_ratio']

        class _Updater(nn.Module):
            def __call__(self, X):
                return (1 - ratio) * X_init + ratio * A @ X[0]

        def _updater(X):
            return (1 - ratio) * X_init + ratio * A @ X, _Updater()
        return _updater
    elif aggregation_model == 'adj':
        class _Updater(nn.Module):
            def __call__(self, X):
                return A @ X[0]

        def _updater(X):
            return A @ X, _Updater()
        return _updater
    elif aggregation_model == 'kta':
        kta_args = train_args['kta_args']
        As = kta.compute_A_powers(A)
        ky = kta.compute_ky(y)
        return lambda X: kta.update_X(
            X, As, ky, indices, kta_args)
    else:
        raise ValueError(f'Invalid aggregation model={aggregation_model}')


def train(dataset, model_args, train_args):
    data, in_features, out_features = dataset
    device = train_args['device']
    data.to(device)
    X, A, y, indices = data[0]
    x_updater = _get_updater(train_args, X, A, y, indices)
    train_idx, val_idx, test_idx = indices
    print(len(train_idx), len(val_idx), len(test_idx))

    n_train = len(train_idx)
    w = torch.ones(n_train).float() / n_train
    pred = torch.zeros((len(X), out_features)).float()

    batchsize = train_args['batchsize']

    def prepare_minibatch(X, y, w):
        idx = np.random.permutation(len(X))[:batchsize]
        return X[idx], y[idx], w[idx].detach().to(device)

    T = train_args['n_weak_learners']
    clip_eps = train_args['clip_eps']

    best_acc = .0, .0, .0
    cosine_array, loss_array, acc_array = [], [], []
    weak_learners, wl_weights, updaters = [], [], []
    for t in range(T):
        # Train
        minibatch = prepare_minibatch(X[train_idx], y[train_idx], w)
        model = model_.Model(
            in_features, model_args['n_units'], out_features,
            model_args['n_layers'], model_args)
        model.to(device)
        model.fit(*minibatch)
        weak_learners.append(model)

        # Update
        pred_one = model.predict(X)
        alpha = _compute_alpha(
            pred_one[train_idx], y[train_idx], w, out_features, clip_eps)
        w = _update_weight(w, alpha, pred_one[train_idx], y[train_idx])
        pred, diff = _update_predict(pred, alpha, pred_one)
        wl_weights.append(alpha)
        X, updater = x_updater(X)
        updaters.append(updater)

        # Evaluate
        (loss_train, loss_val, loss_test), (grad, _, _) = \
            util.compute_loss_and_gradient(pred, y, indices)
        acc_train, acc_val, acc_test = metric.accuracy_all(pred, y, indices)
        cosine_ = util.compute_cosine(
            -grad.flatten(), diff[train_idx].flatten())
        if acc_val > best_acc[1]:
            best_acc = acc_train, acc_val, acc_test

        # Save
        cosine_array.append(cosine_)
        loss_array.append(
            (float(loss_train), float(loss_val), float(loss_test)))
        acc_array.append((acc_train, acc_val, acc_test))

        # Display
        print(f'[Epoch {t}] '
              f'Train Acc: {acc_train:.5f} '
              f'Val Acc: {acc_val:.5f} '
              f'Test Acc: {acc_test:.5f} '
              f'Train Loss: {loss_train:.5f} '
              f'Val Loss: {loss_val:.5f} '
              f'Test Loss: {loss_test:.5f} '
              f'cosine: {cosine_:.5f}')

        yield best_acc[1], t, False, (weak_learners, wl_weights)

    loss_array = np.array(loss_array).T
    acc_array = np.array(acc_array).T

    ret = {
        'acc': {
            'train': best_acc[0],
            'val': best_acc[1],
            'test': best_acc[2],
        },
        'cosine': np.array(cosine_array),
        'acc_all': {
            'train': acc_array[0],
            'val': acc_array[1],
            'test': acc_array[2]
        },
        'loss': {
            'train': loss_array[0],
            'val': loss_array[1],
            'test': loss_array[2]
        }
    }
    yield ret, T, True, (weak_learners, wl_weights, updaters)
