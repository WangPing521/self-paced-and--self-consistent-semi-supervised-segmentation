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
#"python -O main.py seed=213 Data.seed=12 Arch.num_classes=2 Optim.lr=0.00003 Trainer.name=twostage Dataset=spleen num_models=1 Trainer.save_dir=${save_dir}/baseline_p_lr0.00003_run2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py seed=231 Data.seed=12 Arch.num_classes=2 Optim.lr=0.00003 Trainer.name=twostage Dataset=spleen num_models=1 Trainer.save_dir=${save_dir}/baseline_p_lr0.00003_run3 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

#"python -O main.py Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=twostage Dataset=spleen num_models=1 Trainer.save_dir=${save_dir}/baseline_p_lr503_run1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py seed=213 Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=twostage Dataset=spleen num_models=1 Trainer.save_dir=${save_dir}/baseline_p_lr503_run2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py seed=231 Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=twostage Dataset=spleen num_models=1 Trainer.save_dir=${save_dir}/baseline_p_lr503_run3 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

# entropy
#"python -O main.py Data.seed=12 Arch.num_classes=2 Optim.lr=0.00003 Trainer.name=twostage Dataset=spleen num_models=1 StartTraining.entropy_min=True RegScheduler1.max_value=0.0001 Trainer.save_dir=${save_dir}/entropy301 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py seed=213 Data.seed=12 Arch.num_classes=2 Optim.lr=0.00003 Trainer.name=twostage Dataset=spleen num_models=1 StartTraining.entropy_min=True RegScheduler1.max_value=0.0001 Trainer.save_dir=${save_dir}/entropy301_run2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py seed=231 Data.seed=12 Arch.num_classes=2 Optim.lr=0.00003 Trainer.name=twostage Dataset=spleen num_models=1 StartTraining.entropy_min=True RegScheduler1.max_value=0.0001 Trainer.save_dir=${save_dir}/entropy301_run3 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

#"python -O main.py seed=213 Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=twostage Dataset=spleen num_models=1 StartTraining.entropy_min=True RegScheduler1.max_value=0.000005 Trainer.save_dir=${save_dir}/entropy_lr503_505_run2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py seed=213 Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=twostage Dataset=spleen num_models=1 StartTraining.entropy_min=True RegScheduler1.max_value=0.000003 Trainer.save_dir=${save_dir}/entropy_lr503_503_run2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py seed=231 Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=twostage Dataset=spleen num_models=1 StartTraining.entropy_min=True RegScheduler1.max_value=0.000003 Trainer.save_dir=${save_dir}/entropy_lr503_503_run3 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

# MT
#"python -O main.py Data.seed=12 Arch.num_classes=2 Optim.lr=0.00003 Trainer.name=meanteacher usetransform=True Dataset=spleen num_models=2 RegScheduler.max_value=4 Trainer.save_dir=${save_dir}/meanteacher4 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py seed=213 Data.seed=12 Arch.num_classes=2 Optim.lr=0.00003 Trainer.name=meanteacher usetransform=True Dataset=spleen num_models=2 RegScheduler.max_value=4 Trainer.save_dir=${save_dir}/meanteacher4_run2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py seed=231 Data.seed=12 Arch.num_classes=2 Optim.lr=0.00003 Trainer.name=meanteacher usetransform=True Dataset=spleen num_models=2 RegScheduler.max_value=4 Trainer.save_dir=${save_dir}/meanteacher4_run3 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

# test
#"python -O main.py Data.seed=12 Arch.num_classes=2 Optim.lr=0.00003 Trainer.name=uncertaintyMT usetransform=True Dataset=spleen num_models=2 RegScheduler.max_value=4 RegScheduler1.min_value=0.5198 RegScheduler1.max_value=0.7 Trainer.save_dir=${save_dir}/UA-MT Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

# co-training
"python -O main.py Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=twostage StartTraining.train_jsd=True Dataset=spleen num_models=2 RegScheduler.max_value=0.001 AlphaScheduler.begin_epoch=100 Trainer.save_dir=${save_dir}/co-training201_run1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py seed=213 Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=twostage StartTraining.train_jsd=True Dataset=spleen num_models=2 RegScheduler.max_value=0.001 AlphaScheduler.begin_epoch=100 Trainer.save_dir=${save_dir}/co-training201_run2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py seed=231 Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=twostage StartTraining.train_jsd=True Dataset=spleen num_models=2 RegScheduler.max_value=0.001 AlphaScheduler.begin_epoch=100 Trainer.save_dir=${save_dir}/co-training201_run3 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

"python -O main.py Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=twostage StartTraining.train_jsd=True Dataset=spleen num_models=2 RegScheduler.max_value=0.003 AlphaScheduler.begin_epoch=100 Trainer.save_dir=${save_dir}/co-training203_run1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py seed=213 Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=twostage StartTraining.train_jsd=True Dataset=spleen num_models=2 RegScheduler.max_value=0.003 AlphaScheduler.begin_epoch=100 Trainer.save_dir=${save_dir}/co-training203_run2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py seed=231 Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=twostage StartTraining.train_jsd=True Dataset=spleen num_models=2 RegScheduler.max_value=0.003 AlphaScheduler.begin_epoch=100 Trainer.save_dir=${save_dir}/co-training203_run3 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

"python -O main.py Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=twostage StartTraining.train_jsd=True Dataset=spleen num_models=2 RegScheduler.max_value=0.005 AlphaScheduler.begin_epoch=100 Trainer.save_dir=${save_dir}/co-training205_run1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py seed=213 Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=twostage StartTraining.train_jsd=True Dataset=spleen num_models=2 RegScheduler.max_value=0.005 AlphaScheduler.begin_epoch=100 Trainer.save_dir=${save_dir}/co-training205_run2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py seed=231 Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=twostage StartTraining.train_jsd=True Dataset=spleen num_models=2 RegScheduler.max_value=0.005 AlphaScheduler.begin_epoch=100 Trainer.save_dir=${save_dir}/co-training205_run3 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

"python -O main.py Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=twostage StartTraining.train_jsd=True Dataset=spleen num_models=2 RegScheduler.max_value=0.000001 AlphaScheduler.begin_epoch=100 Trainer.save_dir=${save_dir}/co-training501_run1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py seed=213 Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=twostage StartTraining.train_jsd=True Dataset=spleen num_models=2 RegScheduler.max_value=0.000001 AlphaScheduler.begin_epoch=100 Trainer.save_dir=${save_dir}/co-training501_run2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py seed=231 Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=twostage StartTraining.train_jsd=True Dataset=spleen num_models=2 RegScheduler.max_value=0.000001 AlphaScheduler.begin_epoch=100 Trainer.save_dir=${save_dir}/co-training501_run3 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

"python -O main.py Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=twostage StartTraining.train_jsd=True Dataset=spleen num_models=2 RegScheduler.max_value=0.000003 AlphaScheduler.begin_epoch=100 Trainer.save_dir=${save_dir}/co-training503_run1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py seed=213 Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=twostage StartTraining.train_jsd=True Dataset=spleen num_models=2 RegScheduler.max_value=0.000003 AlphaScheduler.begin_epoch=100 Trainer.save_dir=${save_dir}/co-training503_run2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py seed=231 Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=twostage StartTraining.train_jsd=True Dataset=spleen num_models=2 RegScheduler.max_value=0.000003 AlphaScheduler.begin_epoch=100 Trainer.save_dir=${save_dir}/co-training503_run3 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

"python -O main.py Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=twostage StartTraining.train_jsd=True Dataset=spleen num_models=2 RegScheduler.max_value=0.000005 AlphaScheduler.begin_epoch=100 Trainer.save_dir=${save_dir}/co-training505_run1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py seed=213 Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=twostage StartTraining.train_jsd=True Dataset=spleen num_models=2 RegScheduler.max_value=0.000005 AlphaScheduler.begin_epoch=100 Trainer.save_dir=${save_dir}/co-training505_run2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py seed=231 Data.seed=12 Arch.num_classes=2 Optim.lr=0.000003 Trainer.name=twostage StartTraining.train_jsd=True Dataset=spleen num_models=2 RegScheduler.max_value=0.000005 AlphaScheduler.begin_epoch=100 Trainer.save_dir=${save_dir}/co-training505_run3 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

# our

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done
