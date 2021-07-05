#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="../experimental_check/CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=6
account=def-chdesa
save_dir=UA-MT

num_batches=200
ratio1=0.1
unlab_ratio1=$(python -c "print(1-${ratio1})")

declare -a StringArray=(

"python -O main.py Arch.name=unet Trainer.name=meanteacher RegScheduler.max_value=4 Trainer.save_dir=${save_dir}/unet/MT Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Arch.name=unet Trainer.name=uncertaintyMT RegScheduler.max_value=4 RegScheduler1.min_value=1.4 RegScheduler1.max_value=1.4 Trainer.save_dir=${save_dir}/unet/UA_MT_base Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Arch.name=unet Trainer.name=uncertaintyMT RegScheduler.max_value=4 RegScheduler1.min_value=1.039 RegScheduler1.max_value=1.4 Trainer.save_dir=${save_dir}/unet/UA_MT Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

"python -O main.py Arch.name=segnet Trainer.name=meanteacher RegScheduler.max_value=4 Trainer.save_dir=${save_dir}/segnet/MT Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Arch.name=segnet Trainer.name=uncertaintyMT RegScheduler.max_value=4 RegScheduler1.min_value=1.4 RegScheduler1.max_value=1.4 Trainer.save_dir=${save_dir}/segnet/UA_MT_base Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Arch.name=segnet Trainer.name=uncertaintyMT RegScheduler.max_value=4 RegScheduler1.min_value=1.039 RegScheduler1.max_value=1.4 Trainer.save_dir=${save_dir}/segnet/UA_MT Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"


)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done

