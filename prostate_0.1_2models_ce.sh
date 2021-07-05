#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="../experimental_check/CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=5
account=def-chdesa
save_dir=check_prostate

num_batches=200
ratio1=0.1
unlab_ratio1=$(python -c "print(1-${ratio1})")

declare -a StringArray=(
# baseline f
#"python -O main.py Arch.num_classes=2 Optim.lr=0.00001 Dataset=prostate num_models=1 Trainer.save_dir=${save_dir}/baseline_f Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=0.01 Data.labeled_data_ratio=0.99"

# baseline p
#"python -O main.py Arch.num_classes=2 Trainer.name=twostage Optim.lr=0.00001 Dataset=prostate num_models=1 Trainer.save_dir=${save_dir}/baseline_p_lr401 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

# entropy
#"python -O main.py Arch.num_classes=2 Trainer.name=twostage Optim.lr=0.00001 Dataset=prostate num_models=1 StartTraining.entropy_min=True RegScheduler1.max_value=0.001 Trainer.save_dir=${save_dir}/entropy_lr401_201 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

# mt
#"python -O main.py Arch.num_classes=2 Optim.lr=0.00001 Trainer.name=meanteacher usetransform=True Dataset=prostate num_models=2 RegScheduler.max_value=8 Trainer.save_dir=${save_dir}/meanteacher8_run1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py seed=213 Arch.num_classes=2 Optim.lr=0.00001 Trainer.name=meanteacher usetransform=True Dataset=prostate num_models=2 RegScheduler.max_value=8 Trainer.save_dir=${save_dir}/meanteacher8_run2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py seed=231 Arch.num_classes=2 Optim.lr=0.00001 Trainer.name=meanteacher usetransform=True Dataset=prostate num_models=2 RegScheduler.max_value=8 Trainer.save_dir=${save_dir}/meanteacher8_run3 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

# co-training
"python -O main.py Arch.num_classes=2 Optim.lr=0.00001 Dataset=prostate Trainer.name=twostage StartTraining.train_jsd=True num_models=2 RegScheduler.max_value=0.0001 AlphaScheduler.begin_epoch=100 AlphaScheduler.max_value=0.0 Trainer.save_dir=${save_dir}/co_training301_run1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py seed=213 Arch.num_classes=2 Optim.lr=0.00001 Dataset=prostate Trainer.name=twostage StartTraining.train_jsd=True num_models=2 RegScheduler.max_value=0.0001 AlphaScheduler.begin_epoch=100 AlphaScheduler.max_value=0.0 Trainer.save_dir=${save_dir}/co_training301_run2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py seed=231 Arch.num_classes=2 Optim.lr=0.00001 Dataset=prostate Trainer.name=twostage StartTraining.train_jsd=True num_models=2 RegScheduler.max_value=0.0001 AlphaScheduler.begin_epoch=100 AlphaScheduler.max_value=0.0 Trainer.save_dir=${save_dir}/co_training301_run3 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done
