#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="../experimental_check/CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=5
account=def-chdesa
save_dir=acdc_0.1_segnet

num_batches=200
ratio1=0.1
unlab_ratio1=$(python -c "print(1-${ratio1})")

declare -a StringArray=(
"python -O main.py Arch.name=segnet Arch.num_classes=4 Optim.lr=0.0001 num_models=1 Trainer.save_dir=${save_dir}/baseline_f_1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=0.01 Data.labeled_data_ratio=0.99"
"python -O main.py Arch.name=segnet Arch.num_classes=4 Optim.lr=0.0001 num_models=1 Trainer.save_dir=${save_dir}/baseline_p_1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Arch.name=segnet Arch.num_classes=4 Optim.lr=0.0001 usetransform=True Trainer.name=meanteacher RegScheduler.max_value=4 Trainer.save_dir=${save_dir}/meanteacher_1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Arch.name=segnet Arch.num_classes=4 Optim.lr=0.0001 usetransform=True Trainer.name=co_mt num_models=4 RegScheduler.max_value=4 RegScheduler1.max_value=0.5 AlphaScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/our_0.0001_1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

"python -O main.py Arch.name=segnet seed=213 Arch.num_classes=4 Optim.lr=0.0001 num_models=1 Trainer.save_dir=${save_dir}/baseline_f_2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=0.01 Data.labeled_data_ratio=0.99"
"python -O main.py Arch.name=segnet seed=213 Arch.num_classes=4 Optim.lr=0.0001 num_models=1 Trainer.save_dir=${save_dir}/baseline_p_2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Arch.name=segnet seed=213 Arch.num_classes=4 Optim.lr=0.0001 usetransform=True Trainer.name=meanteacher RegScheduler.max_value=4 Trainer.save_dir=${save_dir}/meanteacher_2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Arch.name=segnet seed=213 Arch.num_classes=4 Optim.lr=0.0001 usetransform=True Trainer.name=co_mt num_models=4 RegScheduler.max_value=4 RegScheduler1.max_value=0.5 AlphaScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/our_0.0001_2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

"python -O main.py Arch.name=segnet seed=231 Arch.num_classes=4 Optim.lr=0.0001 num_models=1 Trainer.save_dir=${save_dir}/baseline_f_3 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=0.01 Data.labeled_data_ratio=0.99"
"python -O main.py Arch.name=segnet seed=231 Arch.num_classes=4 Optim.lr=0.0001 num_models=1 Trainer.save_dir=${save_dir}/baseline_p_3 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Arch.name=segnet seed=231 Arch.num_classes=4 Optim.lr=0.0001 usetransform=True Trainer.name=meanteacher RegScheduler.max_value=4 Trainer.save_dir=${save_dir}/meanteacher_3 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Arch.name=segnet seed=231 Arch.num_classes=4 Optim.lr=0.0001 usetransform=True Trainer.name=co_mt num_models=4 RegScheduler.max_value=4 RegScheduler1.max_value=0.5 AlphaScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/our_0.0001_3 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done
