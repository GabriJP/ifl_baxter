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
#printf %s\\n {1..30} | xargs -t -I @ -P "${N_CONCURRENT}" -n 1 train_scripts/execute_federated_fedavg.sh "$W_P" "fedavg_${DATA_NAME}_all" "real_data/${DATA_NAME}" $N_ROUNDS $EPOCHS $MIN_CLIENTS $MIN_FIT_CLIENTS '@' "${N_CONCURRENT}"
#printf %s\\n {1..30} | xargs -t -I @ -P "${N_CONCURRENT}" -n 1 train_scripts/execute_federated_fedcube_noq.sh "$W_P" "fedcube_noq_${DATA_NAME}_all" "real_data/${DATA_NAME}" $N_ROUNDS $EPOCHS $MIN_CLIENTS $MIN_FIT_CLIENTS '@' "${N_CONCURRENT}"
printf %s\\n {1..30} | xargs -t -I @ -P "${N_CONCURRENT}" -n 1 train_scripts/execute_federated_fedprox.sh "$W_P" "fedprox_${DATA_NAME}_all" "real_data/${DATA_NAME}" $N_ROUNDS $EPOCHS $MIN_CLIENTS $MIN_FIT_CLIENTS '@' "${N_CONCURRENT}" 0.75

# n70
MIN_CLIENTS=$(( $(ls real_data/cut_t70_n70/train | wc -l) >> 1 ))
DATA_NAME='cut_t70_n70'
#printf %s\\n {1..30} | xargs -t -I @ -P "${N_CONCURRENT}" -n 1 train_scripts/execute_federated_fedavg.sh "$W_P" "fedavg_${DATA_NAME}" "real_data/${DATA_NAME}" $N_ROUNDS $EPOCHS $MIN_CLIENTS $MIN_FIT_CLIENTS '@' "${N_CONCURRENT}"
#printf %s\\n {1..30} | xargs -t -I @ -P "${N_CONCURRENT}" -n 1 train_scripts/execute_federated_fedcube_noq.sh "$W_P" "fedcube_noq_${DATA_NAME}" "real_data/${DATA_NAME}" $N_ROUNDS $EPOCHS $MIN_CLIENTS $MIN_FIT_CLIENTS '@' "${N_CONCURRENT}"
printf %s\\n {1..30} | xargs -t -I @ -P "${N_CONCURRENT}" -n 1 train_scripts/execute_federated_fedprox.sh "$W_P" "fedprox_${DATA_NAME}" "real_data/${DATA_NAME}" $N_ROUNDS $EPOCHS $MIN_CLIENTS $MIN_FIT_CLIENTS '@' "${N_CONCURRENT}" 0.75

# o70
MIN_CLIENTS=$(( $(ls real_data/cut_t70_o70/train | wc -l) >> 1 ))
DATA_NAME='cut_t70_o70'
#printf %s\\n {1..30} | xargs -t -I @ -P "${N_CONCURRENT}" -n 1 train_scripts/execute_federated_fedavg.sh "$W_P" "fedavg_${DATA_NAME}" "real_data/${DATA_NAME}" $N_ROUNDS $EPOCHS $MIN_CLIENTS $MIN_FIT_CLIENTS '@' "${N_CONCURRENT}"
#printf %s\\n {1..30} | xargs -t -I @ -P "${N_CONCURRENT}" -n 1 train_scripts/execute_federated_fedcube_noq.sh "$W_P" "fedcube_noq_${DATA_NAME}" "real_data/${DATA_NAME}" $N_ROUNDS $EPOCHS $MIN_CLIENTS $MIN_FIT_CLIENTS '@' "${N_CONCURRENT}"
printf %s\\n {1..30} | xargs -t -I @ -P "${N_CONCURRENT}" -n 1 train_scripts/execute_federated_fedprox.sh "$W_P" "fedprox_${DATA_NAME}" "real_data/${DATA_NAME}" $N_ROUNDS $EPOCHS $MIN_CLIENTS $MIN_FIT_CLIENTS '@' "${N_CONCURRENT}" 0.75
