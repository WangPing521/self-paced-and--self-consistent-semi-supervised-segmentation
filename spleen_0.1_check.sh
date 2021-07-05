#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="../experimental_check/CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=5
account=def-chdesa
save_dir=check_spleen

num_batches=200
ratio1=0.1
unlab_ratio1=$(python -c "print(1-${ratio1})")

declare -a StringArray=(

# baseline_f lr=0.00001 0.00003 whatever
"python -O main.py Data.seed=12 Arch.num_classes=2 Optim.lr=0.00003 Trainer.name=twostage Dataset=spleen num_models=1 Trainer.save_dir=${save_dir}/baseline_f_lr Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=0.01 Data.labeled_data_ratio=0.99"

# baseline p
"python -O main.py Data.seed=12 Arch.num_classes=2 Optim.lr=0.00003 Trainer.name=twostage Dataset=spleen num_models=1 Trainer.save_dir=${save_dir}/baseline_p_lr0.00003 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done
