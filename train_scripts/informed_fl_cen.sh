#!/usr/bin/env bash

N_CONCURRENT=2

W_P="informed_fl"
EPOCHS=500
BATCH_SIZE=1024


# All
TEST_PATHS_OPT=(real_data/cut_t70/test/*csv)
TRAIN_PATHS_OPT=(real_data/cut_t70/train/*csv)
printf %s\\n {1..30} | xargs -t -I % -P $N_CONCURRENT -n 1 python cen.py cen --wandb-project="${W_P}" --wandb-group="cen_cut_t70_all" --wandb-name='%' "${TEST_PATHS_OPT[@]/#/--test-paths=}"  "${TRAIN_PATHS_OPT[@]/#/--train-paths=}" --epochs="$EPOCHS" --batch-size="$BATCH_SIZE"
