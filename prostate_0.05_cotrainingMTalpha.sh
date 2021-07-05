#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="../experimental_check/CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=5
account=def-chdesa
save_dir=prostate_0.05_co_trainingalpha

num_batches=200
ratio1=0.05
unlab_ratio1=$(python -c "print(1-${ratio1})")

declare -a StringArray=(

"python -O main.py Dataset=prostate Arch.num_classes=2 Optim.lr=0.00001 Trainer.name=co_mt num_models=4 RegScheduler.max_value=4 RegScheduler1.max_value=0.1 AlphaScheduler.max_value=0.2 Trainer.save_dir=${save_dir}/cotraining_mt_0.2_1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Dataset=prostate Arch.num_classes=2 Optim.lr=0.00001 Trainer.name=co_mt num_models=4 RegScheduler.max_value=4 RegScheduler1.max_value=0.1 AlphaScheduler.max_value=0.1 Trainer.save_dir=${save_dir}/cotraining_mt_0.1_1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

"python -O main.py seed=213 Dataset=prostate Arch.num_classes=2 Optim.lr=0.00001 Trainer.name=co_mt num_models=4 RegScheduler.max_value=4 RegScheduler1.max_value=0.1 AlphaScheduler.max_value=0.2 Trainer.save_dir=${save_dir}/cotraining_mt_0.2_2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py seed=213 Dataset=prostate Arch.num_classes=2 Optim.lr=0.00001 Trainer.name=co_mt num_models=4 RegScheduler.max_value=4 RegScheduler1.max_value=0.1 AlphaScheduler.max_value=0.1 Trainer.save_dir=${save_dir}/cotraining_mt_0.1_2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

"python -O main.py seed=231 Dataset=prostate Arch.num_classes=2 Optim.lr=0.00001 Trainer.name=co_mt num_models=4 RegScheduler.max_value=4 RegScheduler1.max_value=0.1 AlphaScheduler.max_value=0.2 Trainer.save_dir=${save_dir}/cotraining_mt_0.2_3 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py seed=231 Dataset=prostate Arch.num_classes=2 Optim.lr=0.00001 Trainer.name=co_mt num_models=4 RegScheduler.max_value=4 RegScheduler1.max_value=0.1 AlphaScheduler.max_value=0.1 Trainer.save_dir=${save_dir}/cotraining_mt_0.1_3 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"


)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done
