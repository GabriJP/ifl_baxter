#!/usr/bin/env bash

export TF_FORCE_GPU_ALLOW_GROWTH=true

W_P=${1:-"informed_fl"}
GID=${2:-"fedprox"}
DATA_PATH=${3:-"real_data/cut_t70"}
N_ROUNDS=${4:-"500"}
EPOCHS=${5:-"10"}
MIN_CLIENTS=${6:-"4"}
MIN_FIT_CLIENTS=${7-"7"}
JOB_ID=${8:-"0"}
MODULO_N=${9:-"1"}
FEDPROX_PMU=${10:-"0.1"}

sleep "$(( JOB_ID % MODULO_N * MODULO_N ))"

PORT=$(( JOB_ID + 8080 ))

CLIENT_OPTS="xavier:${PORT} --wandb-project='${W_P}' --wandb-group='${GID}' --fedprox-pmu=${FEDPROX_PMU} --light"

# shellcheck disable=SC2086
readarray -t TRAIN_PATHS_OPT < <(ls -S ${DATA_PATH}/train/*csv)
# shellcheck disable=SC2086
readarray -t TEST_PATHS_OPT < <(ls -S ${DATA_PATH}/test/*csv)
CLIENTS=('almogrote' 'almogrote' 'almogrote' 'gofio' 'gofio' 'platano' 'platano')

exec_client() {
  echo "Sending commands to ${CLIENT_NAME}"
  # shellcheck disable=SC2087
  ssh "$CLIENT_NAME" bash <<EOC
cd "\${HOME}/ifl_baxter" || exit
export PATH="\${HOME}/miniconda3/condabin:$PATH"
eval "\$(conda shell.bash hook)"
conda activate baxter_torch || exit
export FLWR_TELEMETRY_ENABLED=0 TF_FORCE_GPU_ALLOW_GROWTH=true KERAS_BACKEND=torch

for TRAIN_PATH_CLIENT in ${TRAIN_PATHS[@]}; do
  TRAIN_NAME="\$(basename "\$TRAIN_PATH_CLIENT")"
  TRAIN_NAME="\${TRAIN_NAME%.*}"
  nohup python fed.py client $CLIENT_OPTS --wandb-name="\${TRAIN_NAME}" ${TEST_PATHS_OPT[@]/#/--test-paths=} --train-paths="\${TRAIN_PATH_CLIENT}" >"${GID}_\${TRAIN_NAME}.log" 2>&1 </dev/null &
done

EOC
}

echo "Starting server"
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
# shellcheck disable=SC2086
nohup python3 fed.py server --port="${PORT}" --wandb-project="${W_P}" --wandb-group="${GID}" --num-rounds=$N_ROUNDS --epochs=$EPOCHS --batch-size=1024 --min-fit-clients=$MIN_FIT_CLIENTS --min-evaluate-clients=1 --min-available-clients=$MIN_CLIENTS --strategy=fedprox --fedprox-pmu=$FEDPROX_PMU >"${W_P}_${GID}_server.log" 2>&1 </dev/null &
echo "Delay"
sleep 25

declare -A CLIENT_MAP

CLIENT_I=0
PATH_I=0
N_CLIENTS_TOTAL=${#CLIENTS[@]}
N_PATHS_LEFT=${#TRAIN_PATHS_OPT[@]}
for CLIENT_NAME in "${CLIENTS[@]}"; do
  N_CLIENTS_LEFT=$((N_CLIENTS_TOTAL-CLIENT_I))
  N_PATHS_CURRENT=$(((N_PATHS_LEFT+N_CLIENTS_LEFT-1) / N_CLIENTS_LEFT))

  for _ in $(seq 1 $N_PATHS_CURRENT) ; do
      TRAIN_PATH=${TRAIN_PATHS_OPT[$PATH_I]}
      CLIENT_MAP[$CLIENT_NAME]="${CLIENT_MAP[$CLIENT_NAME]} ${TRAIN_PATH}"
      PATH_I=$((PATH_I+1))
  done

  CLIENT_I=$((CLIENT_I+1))
  N_PATHS_LEFT=$((N_PATHS_LEFT-N_PATHS_CURRENT))
done

for CLIENT_NAME in "${!CLIENT_MAP[@]}"; do
  TRAIN_PATHS=${CLIENT_MAP[${CLIENT_NAME}]}
  exec_client
done

wait
