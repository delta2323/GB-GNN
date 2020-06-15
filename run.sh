#!/usr/bin/env bash

OPTION=$@

datetime=`date +"%y%m%d_%H%M%S"`
LOG_DIR=results/log/$datetime

mkdir -p $LOG_DIR

if [ -d .git ]; then
    git rev-parse HEAD > $LOG_DIR/git_commit.txt
fi
cp $0 $LOG_DIR/
echo $0 "$@" > $LOG_DIR/command.txt

CMD="python app/main_boosting.py"
PYTHONPATH=$PYTHONPATH:. $CMD $OPTION --seed $RANDOM --out-dir $LOG_DIR > $LOG_DIR/log.txt 2>&1
