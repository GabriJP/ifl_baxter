#!/usr/bin/env bash

export KERAS_BACKEND=torch

N_CONCURRENT=3

N_ROUNDS=500
EPOCHS=10
MIN_FIT_CLIENTS=4

W_P="informed_fl"

CLIENTS=('almogrote' 'gofio' 'platano' 'citic')
for CLIENT_NAME in "${CLIENTS[@]}"; do
  echo "Updating repo on ${CLIENT_NAME}"
  ssh "$CLIENT_NAME" bash <<EOC
cd "\${HOME}/ifl_baxter" || exit
git pull
rm *.log
EOC
done

# All
MIN_CLIENTS=$(( $(ls real_data/cut_t70/train | wc -l) >> 1 ))
DATA_NAME='cut_t70'
for MU in 0.25 0.5 0.75 1.0
do
  printf %s\\n {1..9} | xargs -t -I @ -P "${N_CONCURRENT}" -n 1 train_scripts/execute_federated_fedprox.sh "$W_P" "fedprox_${MU}_${DATA_NAME}_all" "real_data/${DATA_NAME}" $N_ROUNDS $EPOCHS $MIN_CLIENTS $MIN_FIT_CLIENTS '@' "${N_CONCURRENT}" ${MU}
done
