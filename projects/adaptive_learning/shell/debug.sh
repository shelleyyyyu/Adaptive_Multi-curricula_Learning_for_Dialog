#!/usr/bin/env bash

set -e
set -x

FLAG=0

declare -p models=(
  ["seq2seq"]="parlai.agents.adaptive_learning.seq2seq:AdaSeq2seqAgent"
)

declare -p tasks=(
  ["personachat_h3_dynamic_kmeans"]="adaptive_learning:personachat_h3_dynamic_kmeans"
)

declare -p subtasks_list=(
  ["combine_v6"]="kmeans_5:0:4"
)

declare -p bszs=(
  ["seq2seq"]=16 #256
)

declare -p lrs=(
  ["seq2seq"]=5e-4
)

#---------------- main arguments -----------------#
validation_metric_mode=min
validation_metric=ppl
dict_maxtokens=20000
dict_minfreq=-1
reward_metric=total_metric
reward_metric_mode=max
reg_action=0.001
dropout=0.2
#---------------- main arguments -----------------#

#---------------- default arguments --------------#
validation_patience=-1
beam_size=1
report_freq=0.01
anti=False
#---------------- default arguments --------------#

#---------------- model-specific arguments -------#
# ----- RNN encoder-decoder
hiddensize=512
numlayers=2
rnn_class=lstm

# ----- Transformer
n_layers=6
n_heads=8
#---------------- model-specific arguments -------#

function common_args() {
  echo "--validation_metric_mode ${validation_metric_mode} --validation_metric ${validation_metric} --validation_patience ${validation_patience}  --dict_maxtokens ${dict_maxtokens} --dict_minfreq ${dict_minfreq} --beam_size ${beam_size} --report_freq ${report_freq} --dropout ${dropout} --anti ${anti} "$1
}

function train_model() {
  local model_name=$1
  local task_name=$2
  local attr=$3
  local T=$4
  local validation_every_n_secs=$5
  local validation_every_n_epochs=$6
  local num_epochs=$7
  local margin=$8
  local margin_rate=$9

  local model=${models[$model_name]}
  if [[ "${attr}" == "original" ]]; then
    real_task_name=${task_name}_${attr}
  else
    real_task_name=${task_name}
  fi

  local task=${tasks[$real_task_name]}

  if [[ "${attr}" == "loss" ]]; then
    real_attr=loss_of_${model_name}
  else
    real_attr=${attr}
  fi

  local subtasks=${subtasks_list[${real_attr}]}
  #if [[ "${real_attr}" == "combine" ]]; then
  #  subtasks=${subtasks}:loss_of_${model_name}
  #fi

  # shellcheck disable=SC2155
  local model_dir=./models_v2/adaptive_learning_v${FLAG}/"$(hostname)"_gpu${CUDA_VISIBLE_DEVICES}/${model_name}/${task_name}/${real_attr}

  if [[ ! -d "$model_dir" ]]; then
    mkdir -p "${model_dir}"
  fi

  file_name=validby_${validation_metric_mode}_${validation_metric}_per${validation_every_n_secs}secs_per${validation_every_n_epochs}epochs_patience${validation_patience}_dict_maxtokens${dict_maxtokens}_minfreq${dict_minfreq}_bsz${bszs[$model_name]}_beam${beam_size}_${num_epochs}epochs_${dropout}dropout
  # shellcheck disable=SC2155
  local train_args=$(common_args " --model ${model} --task ${task} --subtasks ${subtasks} --learningrate ${lrs[$model_name]} --batchsize ${bszs[$model_name]} --validation_every_n_secs ${validation_every_n_secs} --validation_every_n_epochs ${validation_every_n_epochs}  --num_epochs ${num_epochs} ")

  if [[ "${real_attr}" != "original" ]]; then
    file_name=${file_name}_T${T}_ANTI_${anti}
    train_args=${train_args}" --T ${T}"
  fi

  if [[ "${real_attr}" == "combine" ]]; then
    file_name=${file_name}_teacher_rewardby_${reward_metric_mode}_${reward_metric}_reg_action${reg_action}
    train_args=${train_args}" --reward_metric ${reward_metric} --reward_metric_mode ${reward_metric_mode} --reg_action ${reg_action} "
  fi

  local model_file=${model_dir}/${file_name}
  local train_script=train.py
  train_args=${train_args}" --model_file ${model_file} --tensorboard_comment ${file_name}"

  if [[ "${model_name}" == "seq2seq" ]] || [[ "${model_name}" == "cvae" ]]; then
    train_script=train.py
    train_args=${train_args}" --hiddensize ${hiddensize} --numlayers ${numlayers} --rnn_class ${rnn_class}"
  elif [[ "${model_name}" == "dialogwae" ]] || [[ "${model_name}" == "hred" ]]; then
    train_script=train_dialog_wae.py
    train_args=${train_args}" --hiddensize ${hiddensize} --numlayers ${numlayers} --rnn_class ${rnn_class}"
    if [[ "${model_name}" == "hred" ]]; then
      train_args=${train_args}" --hred True"
    else
      train_args=${train_args}" --hred False"
    fi

  elif [[ "${model_name}" == "transformer" ]]; then
    train_script=train_transformer.py
    train_args=${train_args}" --n_layers ${n_layers} --n_heads ${n_heads}"
  fi
  train_args=${train_args}" --margin ${margin} --margin_rate ${margin_rate}"



  python ./projects/adaptive_learning/${train_script} ${train_args}
  cd -
}

# train_model  MODEL_NAME  TASK_NAME  SUB_TASK  T  VALIDATION_EVERY_N_SECS  VALIDATION_EVERY_N_EPOCHS  NUM_EPOCHS
export CUDA_VISIBLE_DEVICES=-1;
train_model seq2seq personachat_h3_dynamic_kmeans combine_v6 11000 -1 0.2 50 0.5 0.05