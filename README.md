This is the code for the paper titled
Optimization and Generalization Analysis of Transduction through Gradient Boosting and Application to Multi-scale Graph Neural Networks.

# Dependency

- networkx==2.1
- numpy==1.15.1
- optuna==1.3.0
- pytorch-ignite==0.3.0
- scipy==1.1.0
- torch==1.5.0
- pytest==5.0.1 (for testing)

# Preparation

Place `https://github.com/tkipf/gcn/tree/master/gcn/data` as `lib/dataset/data/kipf/` (e.g., `gcn/data/ind.citeseer.allx` should be copied to `lib/dataset/data/kipf/ind.citeseer.allx`.)


# Testing

Unit test
```
PYTHONPATH=$PYTHONPATH:. pytest test/
```

Small experiment test (run on GPU device 0)
```
bash test.sh
```

# Usage

## Commands

### GB-GNN-Adj

```
bash run.sh --device <gpu_id> --dataset <dataset> --min-layer <L_min> --max-layer <L_max> --aggregation-model adj
```

### GB-GNN-Adj + Fine Tuning

```
bash run.sh --device <gpu_id> --dataset <dataset> --min-layer <L_min> --max-layer <L_max> --aggregation-model adj --fine-tune
```

### GB-GNN-KTA

```
bash run.sh --device <gpu_id> --dataset <dataset> --min-layer <L_min> --max-layer <L_max> --aggregation-model kta
```

### GB-GNN-KTA + Fine Tuning

```
bash run.sh --device <gpu_id> --dataset <dataset> --min-layer <L_min> --max-layer <L_max> [--n-weak-learners 40] --aggregation-model kta --fine-tune
```

We set the maximum number of weak learners to 40, as opposed to the default value 100 due to memory constraints in the main paper. To reproduce it, we should set `--n-weak-learners 40`.

### GB-GNN-II

```
bash run.sh --device <gpu_id> --dataset <dataset> --min-layer <L_min> --max-layer <L_max> --aggregation-model ii
```

### GB-GNN-II + Fine Tuning

```
bash run.sh --device <gpu_id> --dataset <dataset> --min-layer <L_min> --max-layer <L_max> --aggregation-model ii --fine-tune
```

## Option
- `<gpu_id>`: GPU ID in use. If we use `-1`, the code runs on CPU
- `<dataset>`: Dataset type. Either `cora`, `citeseer`, or `pubmed` values are allowed.
- `<L_min>`, `<L_max>`: The minimum and maximum number of hidden layer size of the hyperparameter optimization search space. If we want to fix the hidden layer size to `L`, use `L_min=L_max=L`.


## Output

It creates the output directory whose name is the execution time of the form `YYMMDD_HHMMSS`.
The directory has the following files (not a comprehensive list).

- `acc.json`: The accuracies on training, validation, and test datasets.
- `loss/`: The transition of loss values of the best hyperparameter set on training (`train.npy`), validation (`validation.npy`), and test (`test.npy`) datasets
- `cosine.npy`: The transition of cosine values between weak learners and negative gradient on the training dataset.
- `best_params.json`: Chosen hyperparameters.

Accuracies, loss values, and cosine values are for the model with the best hyperparameter set.


# Directory Structures

- `app`: Experiment execution scripts
- `lib`: Implementation of models and their training and evaluation procedures.
- `analysis`: Notebooks for post processing experiment results.
- `test`: Unit test code
