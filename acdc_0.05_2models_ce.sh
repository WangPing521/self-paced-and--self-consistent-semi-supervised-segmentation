#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="../experimental_check/CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=5
account=def-chdesa
save_dir=acdc0.05_cotrainingMT

num_batches=200
ratio1=0.05
unlab_ratio1=$(python -c "print(1-${ratio1})")

declare -a StringArray=(

#"python -O main.py seed=213 Arch.num_classes=4 Optim.lr=0.0001 num_models=1 Trainer.save_dir=${save_dir}/baseline_p Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py seed=213 Arch.num_classes=4 Optim.lr=0.0001 num_models=1 StartTraining.entropy_min=True RegScheduler.max_value=0.00005 Trainer.save_dir=${save_dir}/single_0.00005entropy Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#
#"python -O main.py seed=213 Arch.num_classes=4 Optim.lr=0.0001 usetransform=True StartTraining.train_jsd=True num_models=2 RegScheduler.max_value=0.5 AlphaScheduler.begin_epoch=100 AlphaScheduler.max_value=0.0 Trainer.save_dir=${save_dir}/co_training Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py seed=213 Arch.num_classes=4 Optim.lr=0.0001 usetransform=True Trainer.name=meanteacher  RegScheduler.max_value=4 Trainer.save_dir=${save_dir}/meanteacher Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#
#"python -O main.py seed=213 Arch.num_classes=4 Optim.lr=0.0001 usetransform=True Trainer.name=co_mt num_models=4 RegScheduler.max_value=4 RegScheduler1.max_value=0.5 AlphaScheduler.max_value=0.001 Trainer.save_dir=${save_dir}/our_0.001 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py seed=213 Arch.num_classes=4 Optim.lr=0.0001 usetransform=True Trainer.name=co_mt num_models=4 RegScheduler.max_value=4 RegScheduler1.max_value=0.5 AlphaScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/our_0.0001 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py seed=213 Arch.num_classes=4 Optim.lr=0.0001 usetransform=True Trainer.name=co_mt num_models=4 RegScheduler.max_value=4 RegScheduler1.max_value=0.5 AlphaScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/our_0.00001 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py seed=213 Arch.num_classes=4 Optim.lr=0.0001 usetransform=True Trainer.name=co_mt num_models=4 RegScheduler.max_value=4 RegScheduler1.max_value=0.5 AlphaScheduler.max_value=0.0005 Trainer.save_dir=${save_dir}/our_0.0005 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py seed=213 Arch.num_classes=4 Optim.lr=0.0001 usetransform=True Trainer.name=co_mt num_models=4 RegScheduler.max_value=4 RegScheduler1.max_value=0.5 AlphaScheduler.max_value=0.00005 Trainer.save_dir=${save_dir}/our_0.00005 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py seed=213 Arch.num_classes=4 Optim.lr=0.0001 usetransform=True Trainer.name=co_mt num_models=4 RegScheduler.max_value=4 RegScheduler1.max_value=0.5 AlphaScheduler.max_value=0.000005 Trainer.save_dir=${save_dir}/our_0.000005 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#
#"python -O main.py seed=231 Arch.num_classes=4 Optim.lr=0.0001 num_models=1 Trainer.save_dir=${save_dir}/baseline_p_1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py seed=231 Arch.num_classes=4 Optim.lr=0.0001 num_models=1 StartTraining.entropy_min=True RegScheduler.max_value=0.00005 Trainer.save_dir=${save_dir}/single_0.00005entropy_1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#
#"python -O main.py seed=231 Arch.num_classes=4 Optim.lr=0.0001 usetransform=True StartTraining.train_jsd=True num_models=2 RegScheduler.max_value=0.5 AlphaScheduler.begin_epoch=100 AlphaScheduler.max_value=0.0 Trainer.save_dir=${save_dir}/co_training_1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py seed=231 Arch.num_classes=4 Optim.lr=0.0001 usetransform=True Trainer.name=meanteacher  RegScheduler.max_value=4 Trainer.save_dir=${save_dir}/meanteacher_1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#
#"python -O main.py seed=231 Arch.num_classes=4 Optim.lr=0.0001 usetransform=True Trainer.name=co_mt num_models=4 RegScheduler.max_value=4 RegScheduler1.max_value=0.5 AlphaScheduler.max_value=0.001 Trainer.save_dir=${save_dir}/our_0.001_1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py seed=231 Arch.num_classes=4 Optim.lr=0.0001 usetransform=True Trainer.name=co_mt num_models=4 RegScheduler.max_value=4 RegScheduler1.max_value=0.5 AlphaScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/our_0.0001_1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py seed=231 Arch.num_classes=4 Optim.lr=0.0001 usetransform=True Trainer.name=co_mt num_models=4 RegScheduler.max_value=4 RegScheduler1.max_value=0.5 AlphaScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/our_0.00001_1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py seed=231 Arch.num_classes=4 Optim.lr=0.0001 usetransform=True Trainer.name=co_mt num_models=4 RegScheduler.max_value=4 RegScheduler1.max_value=0.5 AlphaScheduler.max_value=0.0005 Trainer.save_dir=${save_dir}/our_0.0005_1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py seed=231 Arch.num_classes=4 Optim.lr=0.0001 usetransform=True Trainer.name=co_mt num_models=4 RegScheduler.max_value=4 RegScheduler1.max_value=0.5 AlphaScheduler.max_value=0.00005 Trainer.save_dir=${save_dir}/our_0.00005_1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py seed=231 Arch.num_classes=4 Optim.lr=0.0001 usetransform=True Trainer.name=co_mt num_models=4 RegScheduler.max_value=4 RegScheduler1.max_value=0.5 AlphaScheduler.max_value=0.000005 Trainer.save_dir=${save_dir}/our_0.000005_1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

"python -O main.py Arch.num_classes=4 Optim.lr=0.0001 usetransform=True Trainer.name=co_mt num_models=4 RegScheduler.max_value=4 RegScheduler1.max_value=0.5 AlphaScheduler.begin_epoch=100 AlphaScheduler.max_value=0.0 Trainer.save_dir=${save_dir}/cotraining_mt_1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py seed=213 Arch.num_classes=4 Optim.lr=0.0001 usetransform=True Trainer.name=co_mt num_models=4 RegScheduler.max_value=4 RegScheduler1.max_value=0.5 AlphaScheduler.begin_epoch=100 AlphaScheduler.max_value=0.0 Trainer.save_dir=${save_dir}/cotraining_mt_2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py seed=231 Arch.num_classes=4 Optim.lr=0.0001 usetransform=True Trainer.name=co_mt num_models=4 RegScheduler.max_value=4 RegScheduler1.max_value=0.5 AlphaScheduler.begin_epoch=100 AlphaScheduler.max_value=0.0 Trainer.save_dir=${save_dir}/cotraining_mt_3 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done
