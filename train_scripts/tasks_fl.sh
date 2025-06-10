#!/usr/bin/env bash

DATA_NAME=${1:-"not_valid"}

if [ "$DATA_NAME" = "not_valid" ]; then
  echo "One argument required"
  exit 1
fi

N_CONCURRENT=1

export TF_FORCE_GPU_ALLOW_GROWTH=true
W_P="informed_fl_tasks"
N_ROUNDS=500
EPOCHS=10
MIN_FIT_CLIENTS=4

CLIENTS=('almogrote' 'gofio' 'platano' 'citic')
for CLIENT_NAME in "${CLIENTS[@]}"; do
  echo "Updating repo on ${CLIENT_NAME}"
  ssh "$CLIENT_NAME" bash <<EOC
cd "\${HOME}/ifl_baxter" || exit
git pull
rm *.log
EOC
done

MIN_CLIENTS=$(ls -1 real_data/${DATA_NAME}/train/*csv | wc -l)
printf %s\\n {1..30} | xargs -t -I @ -P "${N_CONCURRENT}" -n 1 train_scripts/execute_federated_fedcube_noq.sh "$W_P" "fedcube_noq_${DATA_NAME}" "real_data/${DATA_NAME}" $N_ROUNDS $EPOCHS $MIN_CLIENTS $MIN_FIT_CLIENTS '@' "${N_CONCURRENT}"
