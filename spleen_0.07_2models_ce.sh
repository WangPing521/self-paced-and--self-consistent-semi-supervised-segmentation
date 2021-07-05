#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="../experimental_check/CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=5
account=def-chdesa
save_dir=check_spleen_0.07

num_batches=200

ratio1=0.07
unlab_ratio1=$(python -c "print(1-${ratio1})")

declare -a StringArray=(

# baseline p
#"python -O main.py Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=twostage Dataset=spleen num_models=1 Trainer.save_dir=${save_dir}/baseline_p_lr503 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py seed=213 Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=twostage Dataset=spleen num_models=1 Trainer.save_dir=${save_dir}/baseline_p_lr503_run2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py seed=231 Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=twostage Dataset=spleen num_models=1 Trainer.save_dir=${save_dir}/baseline_p_lr503_run3 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

# meanteacher
#"python -O main.py Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=meanteacher Dataset=spleen usetransform=True num_models=2 RegScheduler.max_value=12 Trainer.save_dir=${save_dir}/meanteacher_lr503_12_run1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py seed=213 Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=meanteacher Dataset=spleen usetransform=True num_models=2 RegScheduler.max_value=12 Trainer.save_dir=${save_dir}/meanteacher_lr503_12_run2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py seed=231 Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=meanteacher Dataset=spleen usetransform=True num_models=2 RegScheduler.max_value=12 Trainer.save_dir=${save_dir}/meanteacher_lr503_12_run3 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

# entropy
#"python -O main.py Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=twostage Dataset=spleen num_models=1 StartTraining.entropy_min=True RegScheduler1.max_value=0.000005 Trainer.save_dir=${save_dir}/entropy_lr503_505_run1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py seed=213 Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=twostage Dataset=spleen num_models=1 StartTraining.entropy_min=True RegScheduler1.max_value=0.000005 Trainer.save_dir=${save_dir}/entropy_lr503_505_run2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py seed=231 Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=twostage Dataset=spleen num_models=1 StartTraining.entropy_min=True RegScheduler1.max_value=0.000005 Trainer.save_dir=${save_dir}/entropy_lr503_505_run3 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

# test
# co-training
#"python -O main.py Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=twostage StartTraining.train_jsd=True Dataset=spleen num_models=2 RegScheduler.max_value=0.00003 AlphaScheduler.begin_epoch=100 Trainer.save_dir=${save_dir}/co-training403_run1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py seed=213 Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=twostage StartTraining.train_jsd=True Dataset=spleen num_models=2 RegScheduler.max_value=0.00003 AlphaScheduler.begin_epoch=100 Trainer.save_dir=${save_dir}/co-training403_run2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py seed=231 Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=twostage StartTraining.train_jsd=True Dataset=spleen num_models=2 RegScheduler.max_value=0.00003 AlphaScheduler.begin_epoch=100 Trainer.save_dir=${save_dir}/co-training403_run3 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

#"python -O main.py Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=twostage StartTraining.train_jsd=True Dataset=spleen num_models=2 RegScheduler.max_value=0.000001 AlphaScheduler.begin_epoch=100 Trainer.save_dir=${save_dir}/co-training501_run1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py seed=213 Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=twostage StartTraining.train_jsd=True Dataset=spleen num_models=2 RegScheduler.max_value=0.000001 AlphaScheduler.begin_epoch=100 Trainer.save_dir=${save_dir}/co-training501_run2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py seed=231 Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=twostage StartTraining.train_jsd=True Dataset=spleen num_models=2 RegScheduler.max_value=0.000001 AlphaScheduler.begin_epoch=100 Trainer.save_dir=${save_dir}/co-training501_run3 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

"python -O main.py Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=twostage StartTraining.train_jsd=True Dataset=spleen num_models=2 RegScheduler.max_value=0.000003 AlphaScheduler.begin_epoch=100 Trainer.save_dir=${save_dir}/co-training503_run1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py seed=213 Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=twostage StartTraining.train_jsd=True Dataset=spleen num_models=2 RegScheduler.max_value=0.000003 AlphaScheduler.begin_epoch=100 Trainer.save_dir=${save_dir}/co-training503_run2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py seed=231 Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=twostage StartTraining.train_jsd=True Dataset=spleen num_models=2 RegScheduler.max_value=0.000003 AlphaScheduler.begin_epoch=100 Trainer.save_dir=${save_dir}/co-training503_run3 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done
