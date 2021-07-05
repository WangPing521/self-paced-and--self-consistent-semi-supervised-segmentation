#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="../experimental_check/CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=5
account=def-chdesa
save_dir=spleen_SP_our0.07

num_batches=200
ratio1=0.07
unlab_ratio1=$(python -c "print(1-${ratio1})")

declare -a StringArray=(

"python -O main.py Optim.lr=0.00001 Trainer.name=CoMTSelf Dataset=spleen Arch.num_classes=2 num_models=4 AlphaScheduler.max_value=0.000015 lam=0.05 RegScheduler.max_value=20 RegScheduler1.max_value=0.5  Trainer.save_dir=${save_dir}/our_0.05 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Optim.lr=0.00001 Trainer.name=CoMTSelf Dataset=spleen Arch.num_classes=2 num_models=4 AlphaScheduler.max_value=0.000015 lam=0.1 RegScheduler.max_value=20 RegScheduler1.max_value=0.5 Trainer.save_dir=${save_dir}/our_0.1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Optim.lr=0.00001 Trainer.name=CoMTSelf Dataset=spleen Arch.num_classes=2 num_models=4 AlphaScheduler.max_value=0.000015 lam=0.2 RegScheduler.max_value=20 RegScheduler1.max_value=0.5 Trainer.save_dir=${save_dir}/our_0.2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Optim.lr=0.00001 Trainer.name=CoMTSelf Dataset=spleen Arch.num_classes=2 num_models=4 AlphaScheduler.max_value=0.000015 lam=0.5 RegScheduler.max_value=20 RegScheduler1.max_value=0.5 Trainer.save_dir=${save_dir}/our_0.5 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Optim.lr=0.00001 Trainer.name=CoMTSelf Dataset=spleen Arch.num_classes=2 num_models=4 AlphaScheduler.max_value=0.000015 lam=0.8 RegScheduler.max_value=20 RegScheduler1.max_value=0.5 Trainer.save_dir=${save_dir}/our_0.8 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Optim.lr=0.00001 Trainer.name=CoMTSelf Dataset=spleen Arch.num_classes=2 num_models=4 AlphaScheduler.max_value=0.000015 lam=1 RegScheduler.max_value=20 RegScheduler1.max_value=0.5 Trainer.save_dir=${save_dir}/our_1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Optim.lr=0.00001 Trainer.name=CoMTSelf Dataset=spleen Arch.num_classes=2 num_models=4 AlphaScheduler.max_value=0.000015 lam=2 RegScheduler.max_value=20 RegScheduler1.max_value=0.5 Trainer.save_dir=${save_dir}/our_2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Optim.lr=0.00001 Trainer.name=CoMTSelf Dataset=spleen Arch.num_classes=2 num_models=4 AlphaScheduler.max_value=0.000015 lam=4 RegScheduler.max_value=20 RegScheduler1.max_value=0.5 Trainer.save_dir=${save_dir}/our_4 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done

