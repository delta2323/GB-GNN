#!/usr/bin/env bash

set -e

if [ -d .git ]; then
    git rev-parse HEAD
fi

for dataset in cora citeseer pubmed
do
    for agg in no_aggregation ii adj kta
    do
	DEFAULT_OPTION="--device 0 --n-trials 2 --n-weak-learners 2"
	PYTHONPATH=$PYTHONPATH:. python app/main_boosting.py ${DEFAULT_OPTION} --dataset ${dataset} --aggregation-model ${agg}
	PYTHONPATH=$PYTHONPATH:. python app/main_boosting.py ${DEFAULT_OPTION} --dataset ${dataset} --aggregation-model ${agg} --fine-tune
    done
done
