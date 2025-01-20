#!/usr/bin/env bash

export TF_FORCE_GPU_ALLOW_GROWTH=true

N_CONCURRENT=5

W_P="informed_fl"
EPOCHS=500
BATCH_SIZE=1024


# All
TEST_PATHS_OPT=(real_data/cut_t70/test/*csv)
TRAIN_PATHS_OPT=(real_data/cut_t70/train/*csv)
printf %s\\n {1..30} | xargs -t -I % -P $N_CONCURRENT -n 1 python cen.py cen --wandb-project="${W_P}" --wandb-group="cen_cut_t70_all" --wandb-name='%' "${TEST_PATHS_OPT[@]/#/--test-paths=}"  "${TRAIN_PATHS_OPT[@]/#/--train-paths=}" --epochs="$EPOCHS" --batch-size="$BATCH_SIZE"

# n70
TEST_PATHS_OPT=(real_data/cut_t70_n70/test/*csv)
TRAIN_PATHS_OPT=(real_data/cut_t70_n70/train/*csv)
printf %s\\n {1..30} | xargs -t -I % -P $N_CONCURRENT -n 1 python cen.py cen --wandb-project="${W_P}" --wandb-group="cen_cut_t70_n70" --wandb-name='%' "${TEST_PATHS_OPT[@]/#/--test-paths=}"  "${TRAIN_PATHS_OPT[@]/#/--train-paths=}" --epochs="$EPOCHS" --batch-size="$BATCH_SIZE"

# o70
TEST_PATHS_OPT=(real_data/cut_t70_o70/test/*csv)
TRAIN_PATHS_OPT=(real_data/cut_t70_o70/train/*csv)
printf %s\\n {1..30} | xargs -t -I % -P $N_CONCURRENT -n 1 python cen.py cen --wandb-project="${W_P}" --wandb-group="cen_cut_t70_o70" --wandb-name='%' "${TEST_PATHS_OPT[@]/#/--test-paths=}"  "${TRAIN_PATHS_OPT[@]/#/--train-paths=}" --epochs="$EPOCHS" --batch-size="$BATCH_SIZE"
