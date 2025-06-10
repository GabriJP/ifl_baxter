#!/usr/bin/env bash

DATA_NAME=${1:-"not_valid"}
TRAIN_MOTION=${2:-"not_valid"}

if [ "$DATA_NAME" = "not_valid" ]; then
  echo "Two arguments required"
  exit 1
fi

if [ "$TRAIN_MOTION" = "not_valid" ]; then
  echo "Two arguments required"
  exit 1
fi

N_CONCURRENT=5

export TF_FORCE_GPU_ALLOW_GROWTH=true
W_P="informed_fl_tasks"
EPOCHS=500
BATCH_SIZE=1024

TEST_PATHS_OPT=(real_data/"${DATA_NAME}"/test/*csv)
if [ "$TRAIN_MOTION" = " " ]; then
  TRAIN_PATHS_OPT=(real_data/six_areas/test/*csv)
  TRAIN_MOTION="squared_circle"
else
  TRAIN_PATHS_OPT=(real_data/six_areas/test/"${TRAIN_MOTION}"*csv)
fi
printf %s\\n {1..30} | xargs -t -I % -P $N_CONCURRENT -n 1 python cen.py cen --wandb-project="${W_P}" --wandb-group="cen_${DATA_NAME}_train_${TRAIN_MOTION}" --wandb-name='%' "${TEST_PATHS_OPT[@]/#/--test-paths=}"  "${TRAIN_PATHS_OPT[@]/#/--train-paths=}" --epochs="$EPOCHS" --batch-size="$BATCH_SIZE"
