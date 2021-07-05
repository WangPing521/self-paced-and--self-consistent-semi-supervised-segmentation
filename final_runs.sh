#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="../experimental_check/CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=5
account=def-chdesa
save_dir=self_consistency

num_batches=200
ratio1=0.1
unlab_ratio1=$(python -c "print(1-${ratio1})")


declare -a StringArray=(
"python -O main.py Optim.lr=0.0001 Trainer.name=CoMTSelf num_models=4 usetransform=True AlphaScheduler.max_value=0.00015 Pacevalue.min_value=0.1 Pacevalue.max_value=1 RegScheduler.max_value=4 RegScheduler1.max_value=0.5 Trainer.save_dir=${save_dir}/MSE_selfcons Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Optim.lr=0.0001 Trainer.name=CoMTSelf num_models=4 usetransform=True AlphaScheduler.max_value=0.00015 Pacevalue.min_value=0.1 Pacevalue.max_value=1 RegScheduler.max_value=4 RegScheduler1.max_value=0.5 cons_term=jsd Trainer.save_dir=${save_dir}/jsd_selfcons Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done

