#!/usr/bin/env bash

N_CLIENTS=${1:-"0"}
PORT=${2:-"8081"}

if [ "${N_CLIENTS}" -eq 0 ]; then
  echo "N clients not provided"
  exit 1
fi

export TF_FORCE_GPU_ALLOW_GROWTH=true

W_P="ifl_scalability"
DATA_PATH="real_data/cut_t70"
DPT="${DATA_PATH}/train"
N_ROUNDS="500"
EPOCHS="10"
GID="${N_CLIENTS}_clients"

CLIENT_OPTS="xavier:${PORT} --wandb-project='${W_P}' --wandb-group='${GID}'"

readarray -t TEST_PATHS_OPT < <(ls -S ${DATA_PATH}/test/*csv)

for _ in {1..30} ; do
  echo "Starting server"
  export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
  python3 fed.py server --port="${PORT}" --wandb-project="${W_P}" --wandb-group="${GID}" --num-rounds=$N_ROUNDS --epochs=$EPOCHS --batch-size=1024 --min-fit-clients=3 --min-evaluate-clients=1 --min-available-clients=$N_CLIENTS --strategy=fedcube_noq >"${W_P}_${N_CLIENTS}_server.log" 2>&1 </dev/null &
  echo "Delay"
  sleep 25

  . "train_scripts/scalability_${N_CLIENTS}_clients.sh"

  wait
done
