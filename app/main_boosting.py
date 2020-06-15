import argparse
from distutils.util import strtobool
import json
import os
from pathlib import Path

import numpy as np
import optuna
from optuna import exceptions
from optuna import pruners
import torch

import lib.dataset.dataset as D
from lib.dataset import binarize
from lib.boosting import samme
from lib import util
from lib.boosting import fine_tune


parser = argparse.ArgumentParser(description='molnet example')
parser.add_argument('--seed', '-s', type=int, default=0)
# Dataset
parser.add_argument('--dataset', '-d', type=str, default='citeseer',
                    help='Dataset name',
                    choices=('citeseer', 'cora', 'pubmed'))
parser.add_argument('--noisy-graph', '-n', action='store_true',
                    help='Randomly add noise edges to a graph')
parser.add_argument('--binarize', default='False', type=strtobool)
# Optuna
parser.add_argument('--n-trials', default=1000, type=int)
# Model
parser.add_argument('--n-weak-learners', default=100, type=int)
parser.add_argument('--n-iters', default=100, type=int)
parser.add_argument('--batchsize', default=128, type=int)
parser.add_argument('--device', default=-1, type=int)
parser.add_argument('--min-layers', default=0, type=int)
parser.add_argument('--max-layers', default=5, type=int)
parser.add_argument('--aggregation-model', default='adj',
                    choices=('no_aggregation', 'ii', 'adj', 'kta'))
parser.add_argument('--fine-tune', action='store_true')
# IO
parser.add_argument('--out-dir', default='results', type=str)

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

dataset = D.get_dataset(args.dataset, True, True, args.noisy_graph)
if args.binarize:
    dataset = binarize.binarize(dataset)


def _prepare_arguments(trial, args):
    device = util.get_device(args.device)
    n_iters = trial.suggest_int('n_iters', 1, args.n_iters)
    N = len(dataset[0].X)
    batchsize = trial.suggest_int(
        'batchsize', 1, min(args.batchsize, N))
    optimizer = trial.suggest_categorical(
        'optimizer', ('sgd', 'adam', 'rmsprop'))
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    momentum = trial.suggest_loguniform('momentum', 1e-10, 1e-1)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-1)
    n_units = trial.suggest_int('n_units', 10, 200)
    n_layers = trial.suggest_int(
        'n_layers', args.min_layers, args.max_layers)
    dropout = trial.suggest_categorical('dropout', (True, False))
    n_weak_learners = trial.suggest_int(
        'n_weak_learners', 1, args.n_weak_learners)
    clip_eps = trial.suggest_loguniform('clip_eps', 1e-10, 1e-5)

    model_args = {
        'n_iters': n_iters,
        'weight_decay': weight_decay,
        'optimizer': optimizer,
        'lr': lr,
        'momentum': momentum,
        'n_units': n_units,
        'n_layers': n_layers,
        'dropout': dropout,
    }

    train_args = {
        'n_weak_learners': n_weak_learners,
        'batchsize': batchsize,
        'device': device,
        'clip_eps': clip_eps,
        'aggregation_model': args.aggregation_model
    }

    if args.aggregation_model == 'ii':
        ii_aggregation_ratio = trial.suggest_uniform(
            'aggregation_ratio', 0., 1.)
        ii_args = {
            'aggregation_ratio': ii_aggregation_ratio
        }
        train_args['ii_args'] = ii_args

    if args.aggregation_model == 'kta':
        kta_optimizer = trial.suggest_categorical(
            'kta_optimizer', ('sgd', 'adam', 'rmsprop'))
        kta_lr = trial.suggest_loguniform('kta_lr', 1e-5, 1e-1)
        kta_n_epochs = trial.suggest_int('kta_n_epochs', 5, 30)
        kta_args = {
            'optimizer': kta_optimizer,
            'lr': kta_lr,
            'n_epochs': kta_n_epochs,
            'weight_init_method': 'all_one'
        }
        train_args['kta_args'] = kta_args

    if args.fine_tune:
        fine_tune_optimizer = trial.suggest_categorical(
            'fine_tune_optimizer', ('sgd', 'adam', 'rmsprop'))
        fine_tune_lr = trial.suggest_loguniform('fine_tune_lr', 1e-5, 1e-1)
        fine_tune_momentum = trial.suggest_loguniform(
            'fine_tune_momentum', 1e-10, 1e-1)
        fine_tune_weight_decay = trial.suggest_loguniform(
            'fine_tune_weight_decay', 1e-10, 1e-1)
        fine_tune_n_iters = trial.suggest_int('n_iters', 1, args.n_iters)

        fine_tune_args = {
            'optimizer': fine_tune_optimizer,
            'lr': fine_tune_lr,
            'momentum': fine_tune_momentum,
            'weight_decay': fine_tune_weight_decay,
            'n_iters': fine_tune_n_iters,
            'device': device,
        }
        train_args['fine_tune_args'] = fine_tune_args

    return model_args, train_args


def _train(trial, dataset, model_args, train_args):
    for output, step, completed, models in samme.train(
            dataset, model_args, train_args):
        if completed:
            break
        trial.report(output, step)
        if trial.should_prune():
            raise exceptions.TrialPruned()
    return output, models


def _attach(trial, output):
    trial.set_user_attr('acc_train', output['acc']['train'])
    trial.set_user_attr('acc_val', output['acc']['val'])
    trial.set_user_attr('acc_test', output['acc']['test'])
    trial.set_user_attr('cosine', output['cosine'])
    trial.set_user_attr('loss_train', output['loss']['train'])
    trial.set_user_attr('loss_val', output['loss']['val'])
    trial.set_user_attr('loss_test', output['loss']['test'])
    trial.set_user_attr('acc_all_train', output['acc_all']['train'])
    trial.set_user_attr('acc_all_val', output['acc_all']['val'])
    trial.set_user_attr('acc_all_test', output['acc_all']['test'])


def objective(trial, args):
    model_args, train_args = _prepare_arguments(trial, args)
    output, models = _train(trial, dataset, model_args, train_args)

    if args.fine_tune:
        output_fine_tune = fine_tune.fine_tune(
            models, dataset, train_args['fine_tune_args'])
        print(f"Train:\t{output['acc']['train']:.5f}->{output_fine_tune['acc']['train']:.5f}")
        print(f"Val:\t{output['acc']['val']:.5f}->{output_fine_tune['acc']['val']:.5f}")
        print(f"Test:\t{output['acc']['test']:.5f}->{output_fine_tune['acc']['test']:.5f}")
        output = output_fine_tune

    _attach(trial, output)
    return output['acc']['val']


# Optimize
study = optuna.create_study(
    direction='maximize', pruner=pruners.MedianPruner())
study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)


def _save(out_dir, study):
    os.makedirs(out_dir, exist_ok=True)
    out_dir = Path(out_dir)

    best_params = study.best_trial.params
    print('Best Trial: ', best_params)
    with open(out_dir / 'best_params.json', 'w') as f:
        json.dump(best_params, f)

    attrs = study.best_trial.user_attrs
    accs = {
        'train': attrs['acc_train'],
        'val': attrs['acc_val'],
        'test': attrs['acc_test']
    }
    print('Accs: ', accs)
    with open(out_dir / 'acc.json', 'w') as f:
        json.dump(accs, f)

    np.save(out_dir / 'cosine.npy', attrs['cosine'])

    loss_dir = out_dir / 'loss'
    os.makedirs(loss_dir, exist_ok=True)
    np.save(loss_dir / 'train.npy', attrs['loss_train'])
    np.save(loss_dir / 'val.npy', attrs['loss_val'])
    np.save(loss_dir / 'test.npy', attrs['loss_test'])

    acc_dir = out_dir / 'acc_all'
    os.makedirs(acc_dir, exist_ok=True)
    np.save(acc_dir / 'train.npy', attrs['acc_all_train'])
    np.save(acc_dir / 'val.npy', attrs['acc_all_val'])
    np.save(acc_dir / 'test.npy', attrs['acc_all_test'])

    print('Args: ', args)
    with open(out_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f)


_save(args.out_dir, study)
