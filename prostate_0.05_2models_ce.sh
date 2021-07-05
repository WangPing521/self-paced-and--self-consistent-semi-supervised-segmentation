#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="../experimental_check/CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=5
account=def-chdesa
save_dir=check_prostate_0.05

num_batches=200
ratio1=0.05
unlab_ratio1=$(python -c "print(1-${ratio1})")

declare -a StringArray=(
# baseline p
#"python -O main.py Arch.num_classes=2 Optim.lr=0.0001 Dataset=prostate num_models=1 Trainer.name=twostage Trainer.save_dir=${save_dir}/baseline_p_lr301_run1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py seed=213 Arch.num_classes=2 Optim.lr=0.0001 Dataset=prostate num_models=1 Trainer.name=twostage Trainer.save_dir=${save_dir}/baseline_p_lr301_run2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py seed=231 Arch.num_classes=2 Optim.lr=0.0001 Dataset=prostate num_models=1 Trainer.name=twostage Trainer.save_dir=${save_dir}/baseline_p_lr301_run3 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

# entropy
#"python -O main.py Arch.num_classes=2 Optim.lr=0.0001 Dataset=prostate Trainer.name=twostage num_models=1 StartTraining.entropy_min=True RegScheduler1.max_value=0.00001 Trainer.save_dir=${save_dir}/entropy401_run1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py seed=213 Arch.num_classes=2 Optim.lr=0.0001 Dataset=prostate Trainer.name=twostage num_models=1 StartTraining.entropy_min=True RegScheduler1.max_value=0.00001 Trainer.save_dir=${save_dir}/entropy401_run2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py Arch.num_classes=2 Optim.lr=0.0001 Dataset=prostate Trainer.name=twostage num_models=1 StartTraining.entropy_min=True RegScheduler1.max_value=0.00005 Trainer.save_dir=${save_dir}/entropy405_run1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

#co-training
"python -O main.py Arch.num_classes=2 Optim.lr=0.0001 Trainer.name=twostage Dataset=prostate StartTraining.train_jsd=True num_models=2 RegScheduler.max_value=0.0000003 AlphaScheduler.begin_epoch=100 AlphaScheduler.max_value=0.0 Trainer.save_dir=${save_dir}/co_training603_run1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py seed=213 Arch.num_classes=2 Optim.lr=0.0001 Trainer.name=twostage Dataset=prostate StartTraining.train_jsd=True num_models=2 RegScheduler.max_value=0.0000003 AlphaScheduler.begin_epoch=100 AlphaScheduler.max_value=0.0 Trainer.save_dir=${save_dir}/co_training603_run2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py seed=231 Arch.num_classes=2 Optim.lr=0.0001 Trainer.name=twostage Dataset=prostate StartTraining.train_jsd=True num_models=2 RegScheduler.max_value=0.0000003 AlphaScheduler.begin_epoch=100 AlphaScheduler.max_value=0.0 Trainer.save_dir=${save_dir}/co_training603_run3 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"


)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done
