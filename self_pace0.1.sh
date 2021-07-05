#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="../experimental_check/CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=5
account=def-chdesa
save_dir=spleen_SP_0.1

num_batches=200
ratio1=0.1
unlab_ratio1=$(python -c "print(1-${ratio1})")

declare -a StringArray=(

"python -O main.py Optim.lr=0.00001 Trainer.name=selfpace Dataset=spleen Arch.num_classes=2 RegScheduler.max_value=0.000001 AlphaScheduler.max_value=0.000012 lam=0.05 Trainer.save_dir=${save_dir}/SPJSD_0.05 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Optim.lr=0.00001 Trainer.name=selfpace Dataset=spleen Arch.num_classes=2 RegScheduler.max_value=0.000001 AlphaScheduler.max_value=0.000012 lam=0.1 Trainer.save_dir=${save_dir}/SPJSD_0.1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Optim.lr=0.00001 Trainer.name=selfpace Dataset=spleen Arch.num_classes=2 RegScheduler.max_value=0.000001 AlphaScheduler.max_value=0.000012 lam=0.2 Trainer.save_dir=${save_dir}/SPJSD_0.2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Optim.lr=0.00001 Trainer.name=selfpace Dataset=spleen Arch.num_classes=2 RegScheduler.max_value=0.000001 AlphaScheduler.max_value=0.000012 lam=0.5 Trainer.save_dir=${save_dir}/SPJSD_0.5 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Optim.lr=0.00001 Trainer.name=selfpace Dataset=spleen Arch.num_classes=2 RegScheduler.max_value=0.000001 AlphaScheduler.max_value=0.000012 lam=0.8 Trainer.save_dir=${save_dir}/SPJSD_0.8 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Optim.lr=0.00001 Trainer.name=selfpace Dataset=spleen Arch.num_classes=2 RegScheduler.max_value=0.000001 AlphaScheduler.max_value=0.000012 lam=1 Trainer.save_dir=${save_dir}/SPJSD_1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Optim.lr=0.00001 Trainer.name=selfpace Dataset=spleen Arch.num_classes=2 RegScheduler.max_value=0.000001 AlphaScheduler.max_value=0.000012 lam=2 Trainer.save_dir=${save_dir}/SPJSD_2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Optim.lr=0.00001 Trainer.name=selfpace Dataset=spleen Arch.num_classes=2 RegScheduler.max_value=0.000001 AlphaScheduler.max_value=0.000012 lam=4 Trainer.save_dir=${save_dir}/SPJSD_4 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done

