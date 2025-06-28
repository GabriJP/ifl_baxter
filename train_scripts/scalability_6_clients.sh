exec_client() {
  echo "Sending commands to ${CLIENT_NAME}"
  # shellcheck disable=SC2087
  ssh "$CLIENT_NAME" bash <<EOC
cd "\${HOME}/ifl_baxter" || exit
export PATH="\${HOME}/miniconda3/condabin:$PATH"
eval "\$(conda shell.bash hook)"
conda activate baxter_keras || exit
export FLWR_TELEMETRY_ENABLED=0 TF_FORCE_GPU_ALLOW_GROWTH=true

nohup python fed.py client $CLIENT_OPTS --wandb-name="${TRAIN_NAME}" ${TEST_PATHS_OPT[@]/#/--test-paths=} ${TRAIN_PATHS[@]/#/--train-paths=} >"${GID}_${TRAIN_NAME}.log" 2>&1 </dev/null &
EOC
}

CLIENT_NAME=almogrote
TRAIN_NAME="almogrote_1"
TRAIN_PATHS=(
"${DPT}/random_p-15_t105_pos_vel_act_torq_cmd_pp.csv"
"${DPT}/spiral_p-15_t105_pos_vel_act_torq_cmd_pp.csv"
"${DPT}/spiral_p60_t70_pos_vel_act_torq_cmd_pp_1.csv"
)
exec_client
TRAIN_NAME="almogrote_2"
TRAIN_PATHS=(
"${DPT}/spiral_p60_t140_pos_vel_act_torq_cmd_pp.csv"
"${DPT}/random_p15_t70_pos_vel_act_torq_cmd_pp_1.csv"
"${DPT}/random_p15_t70_pos_vel_act_torq_cmd_pp_2.csv"
)
exec_client
CLIENT_NAME=gofio
TRAIN_NAME="gofio_1"
TRAIN_PATHS=(
"${DPT}/random_p90_t105_pos_vel_act_torq_cmd_pp.csv"
"${DPT}/spiral_p90_t105_pos_vel_act_torq_cmd_pp.csv"
"${DPT}/spiral_p60_t70_pos_vel_act_torq_cmd_pp_3.csv"
)
exec_client
TRAIN_NAME="gofio_2"
TRAIN_PATHS=(
"${DPT}/random_p15_t140_pos_vel_act_torq_cmd_pp.csv"
"${DPT}/spiral_p15_t70_pos_vel_act_torq_cmd_pp_1.csv"
"${DPT}/spiral_p15_t70_pos_vel_act_torq_cmd_pp_2.csv"
)
exec_client
CLIENT_NAME=platano
TRAIN_NAME="platano_1"
TRAIN_PATHS=(
"${DPT}/spiral_p15_t140_pos_vel_act_torq_cmd_pp.csv"
"${DPT}/random_p60_t70_pos_vel_act_torq_cmd_pp_1.csv"
"${DPT}/random_p60_t70_pos_vel_act_torq_cmd_pp_2.csv"
)
exec_client
CLIENT_NAME=citic
TRAIN_NAME="citic_1"
TRAIN_PATHS=(
"${DPT}/random_p60_t140_pos_vel_act_torq_cmd_pp.csv"
"${DPT}/spiral_p60_t70_pos_vel_act_torq_cmd_pp_2.csv"
"${DPT}/spiral_p15_t70_pos_vel_act_torq_cmd_pp_3.csv"
)
exec_client
