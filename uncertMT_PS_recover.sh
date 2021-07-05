#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="../experimental_check/CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=6
account=def-chdesa
save_dir=uncertaintyMT_PS_recover

num_batches=200
ratio1=0.1
unlab_ratio1=$(python -c "print(1-${ratio1})")

ratio2=0.05
unlab_ratio2=$(python -c "print(1-${ratio2})")

ratio3=0.07
unlab_ratio3=$(python -c "print(1-${ratio3})")

declare -a StringArray=(

"python -O main.py Arch.num_classes=2 Dataset=prostate Optim.lr=0.00001 Trainer.name=uncertaintyMT num_models=2 RegScheduler.max_value=4 RegScheduler1.min_value=0.5198 RegScheduler1.max_value=0.7 Triainer.checkpoint_path=runs/${save_dir}/0.1/uncertainty_MT_run1 Trainer.save_dir=${save_dir}/0.1/uncertainty_MT_run1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Arch.num_classes=2 Dataset=prostate Optim.lr=0.00001 Trainer.name=uncertaintyMT num_models=2 RegScheduler.max_value=4 RegScheduler1.min_value=0.5198 RegScheduler1.max_value=0.7 Triainer.checkpoint_path=runs/${save_dir}/0.05/uncertainty_MT_run1 Trainer.save_dir=${save_dir}/0.05/uncertainty_MT_run1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio2} Data.labeled_data_ratio=${ratio2}"

"python -O main.py Arch.num_classes=2 Dataset=spleen Optim.lr=0.0001 usetransform=True Trainer.name=uncertaintyMT num_models=2 RegScheduler.max_value=12 RegScheduler1.min_value=0.5198 RegScheduler1.max_value=0.7 Triainer.checkpoint_path=runs/${save_dir}/0.1/Spleen_uncertainty_MT_run1 Trainer.save_dir=${save_dir}/0.1/Spleen_uncertainty_MT_run1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Arch.num_classes=2 Dataset=spleen Optim.lr=0.000001 Trainer.name=uncertaintyMT num_models=2 RegScheduler.max_value=20 RegScheduler1.min_value=0.5198 RegScheduler1.max_value=0.7 Triainer.checkpoint_path=runs/${save_dir}/0.07/Spleen_uncertainty_MT_run1 Trainer.save_dir=${save_dir}/0.07/Spleen_uncertainty_MT_run1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio3} Data.labeled_data_ratio=${ratio3}"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done

