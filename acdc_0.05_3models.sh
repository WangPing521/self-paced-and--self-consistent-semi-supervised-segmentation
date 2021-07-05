#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="../experimental_check/CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=8
account=def-chdesa
save_dir=acdc_0.05_3models_kl_dice_seed

num_batches=200
ratio1=0.05
unlab_ratio1=$(python -c "print(1-${ratio1})")

declare -a StringArray=(

"python -O main.py seed=124 num_models=3 supvised_term=kl_loss StartTraining.train_jsd=True RegScheduler.max_value=0.4 AlphaScheduler.begin_epoch=120 AlphaScheduler.max_value=0.0 Trainer.save_dir=${save_dir}/baseline_jsd_kl Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py seed=124 num_models=3 supvised_term=kl_loss StartTraining.train_jsd=True RegScheduler.max_value=0.4 AlphaScheduler.max_value=0.0001  Trainer.save_dir=${save_dir}/0.0001_kl  Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py seed=124 num_models=3 supvised_term=kl_loss StartTraining.train_jsd=True RegScheduler.max_value=0.4 AlphaScheduler.max_value=0.0005  Trainer.save_dir=${save_dir}/0.0005_kl  Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py seed=124 num_models=3 supvised_term=kl_loss StartTraining.train_jsd=True RegScheduler.max_value=0.4 AlphaScheduler.max_value=0.001 Trainer.save_dir=${save_dir}/0.001_kl  Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py seed=124 num_models=3 supvised_term=kl_loss StartTraining.train_jsd=True RegScheduler.max_value=0.4 AlphaScheduler.max_value=0.01 Trainer.save_dir=${save_dir}/0.01_kl  Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

"python -O main.py seed=124 num_models=3 supvised_term=dice_loss StartTraining.train_jsd=True RegScheduler.max_value=0.4 AlphaScheduler.begin_epoch=120 AlphaScheduler.max_value=0.0 Trainer.save_dir=${save_dir}/baseline_jsd_dice Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py seed=124 num_models=3 supvised_term=dice_loss StartTraining.train_jsd=True RegScheduler.max_value=0.4 AlphaScheduler.max_value=0.0001  Trainer.save_dir=${save_dir}/0.0001_dice  Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py seed=124 num_models=3 supvised_term=dice_loss StartTraining.train_jsd=True RegScheduler.max_value=0.4 AlphaScheduler.max_value=0.0005  Trainer.save_dir=${save_dir}/0.0005_dice  Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py seed=124 num_models=3 supvised_term=dice_loss StartTraining.train_jsd=True RegScheduler.max_value=0.4 AlphaScheduler.max_value=0.001 Trainer.save_dir=${save_dir}/0.001_dice  Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py seed=124 num_models=3 supvised_term=dice_loss StartTraining.train_jsd=True RegScheduler.max_value=0.4 AlphaScheduler.max_value=0.01 Trainer.save_dir=${save_dir}/0.01_dice  Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done
