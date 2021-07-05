#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="../experimental_check/CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=8
account=def-chdesa
save_dir=acdc_0.1_3views

num_batches=200
ratio1=0.1
unlab_ratio1=$(python -c "print(1-${ratio1})")

declare -a StringArray=(

"python -O main.py Arch.num_classes=4 Optim.lr=0.0001 usetransform=True StartTraining.train_jsd=True num_models=3 RegScheduler.max_value=0.1 AlphaScheduler.begin_epoch=100 AlphaScheduler.max_value=0.0 Trainer.save_dir=${save_dir}/co_training_1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Arch.num_classes=4 Optim.lr=0.0001 usetransform=True Trainer.name=co_mt num_models=6 RegScheduler.max_value=4 RegScheduler1.max_value=0.1 AlphaScheduler.max_value=0.0005 Trainer.save_dir=${save_dir}/our_0.0005_1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

"python -O main.py seed=213 Arch.num_classes=4 Optim.lr=0.0001 usetransform=True StartTraining.train_jsd=True num_models=3 RegScheduler.max_value=0.1 AlphaScheduler.begin_epoch=100 AlphaScheduler.max_value=0.0 Trainer.save_dir=${save_dir}/co_training_2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py seed=213 Arch.num_classes=4 Optim.lr=0.0001 usetransform=True Trainer.name=co_mt num_models=6 RegScheduler.max_value=4 RegScheduler1.max_value=0.1 AlphaScheduler.max_value=0.0005 Trainer.save_dir=${save_dir}/our_0.0005_2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

"python -O main.py seed=231 Arch.num_classes=4 Optim.lr=0.0001 usetransform=True StartTraining.train_jsd=True num_models=3 RegScheduler.max_value=0.1 AlphaScheduler.begin_epoch=100 AlphaScheduler.max_value=0.0 Trainer.save_dir=${save_dir}/co_training_3 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py seed=231 Arch.num_classes=4 Optim.lr=0.0001 usetransform=True Trainer.name=co_mt num_models=6 RegScheduler.max_value=4 RegScheduler1.max_value=0.1 AlphaScheduler.max_value=0.0005 Trainer.save_dir=${save_dir}/our_0.0005_3 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done
