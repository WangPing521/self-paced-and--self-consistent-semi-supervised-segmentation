#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="../experimental_check/CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=3
account=def-chdesa
save_dir=single_network_segmentation_our

num_batches=200

ratio1=0.1
unlab_ratio1=$(python -c "print(1-${ratio1})")

ratio2=0.05
unlab_ratio2=$(python -c "print(1-${ratio2})")

ratio3=0.07
unlab_ratio3=$(python -c "print(1-${ratio3})")

declare -a StringArray=(

# acdc
"python -O main.py num_models=4 usetransform=True Dataset=acdc Arch.num_classes=4 Optim.lr=0.0001 Trainer.name=CoMTSelf AlphaScheduler.max_value=0.00001 RegScheduler.max_value=4 RegScheduler1.max_value=0.5 Pacevalue.min_value=0.2 Pacevalue.max_value=8 Trainer.checkpoint_path=runs/${save_dir}/acdc_our01_run1 Trainer.save_dir=${save_dir}/acdc_our01_run1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py num_models=4 seed=213 usetransform=True Dataset=acdc Arch.num_classes=4 Optim.lr=0.0001 Trainer.name=CoMTSelf AlphaScheduler.max_value=0.00001 RegScheduler.max_value=4 RegScheduler1.max_value=0.5 Pacevalue.min_value=0.2 Pacevalue.max_value=8 Trainer.checkpoint_path=runs/${save_dir}/acdc_our01_run2 Trainer.save_dir=${save_dir}/acdc_our01_run2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py num_models=4 seed=231 usetransform=True Dataset=acdc Arch.num_classes=4 Optim.lr=0.0001 Trainer.name=CoMTSelf AlphaScheduler.max_value=0.00001 RegScheduler.max_value=4 RegScheduler1.max_value=0.5 Pacevalue.min_value=0.2 Pacevalue.max_value=8 Trainer.checkpoint_path=runs/${save_dir}/acdc_our01_run3 Trainer.save_dir=${save_dir}/acdc_our01_run3 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

# prostate
"python -O main.py num_models=4 usetransform=True Dataset=prostate Trainer.name=CoMTSelf Arch.num_classes=2 Optim.lr=0.00001 AlphaScheduler.max_value=0.2 RegScheduler.max_value=4 RegScheduler1.max_value=0.1 Pacevalue.min_value=0.4 Pacevalue.max_value=4 Trainer.checkpoint_path=runs/${save_dir}/prostate_our01_run1 Trainer.save_dir=${save_dir}/prostate_our01_run1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py num_models=4 usetransform=True seed=213 Dataset=prostate Trainer.name=CoMTSelf Arch.num_classes=2 Optim.lr=0.00001 AlphaScheduler.max_value=0.2 RegScheduler.max_value=4 RegScheduler1.max_value=0.1 Pacevalue.min_value=0.4 Pacevalue.max_value=4 Trainer.checkpoint_path=runs/${save_dir}/prostate_our01_run2 Trainer.save_dir=${save_dir}/prostate_our01_run2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py num_models=4 usetransform=True seed=231 Dataset=prostate Trainer.name=CoMTSelf Arch.num_classes=2 Optim.lr=0.00001 AlphaScheduler.max_value=0.2 RegScheduler.max_value=4 RegScheduler1.max_value=0.1 Pacevalue.min_value=0.4 Pacevalue.max_value=4 Trainer.checkpoint_path=runs/${save_dir}/prostate_our01_run3 Trainer.save_dir=${save_dir}/prostate_our01_run3 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

#spleen
"python -O main.py num_models=4 usetransform=True Dataset=spleen Trainer.name=CoMTSelf Arch.num_classes=2 Optim.lr=0.000003 Data.seed=12 AlphaScheduler.max_value=0.000015 RegScheduler.max_value=12 RegScheduler1.max_value=0.5 Pacevalue.min_value=0.2 Pacevalue.max_value=8 Trainer.checkpoint_path=runs/${save_dir}/spleen_our01_run1 Trainer.save_dir=${save_dir}/spleen_our01_run1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py num_models=4 seed=213 usetransform=True Dataset=spleen Trainer.name=CoMTSelf Arch.num_classes=2 Optim.lr=0.000003 Data.seed=12 AlphaScheduler.max_value=0.000015 RegScheduler.max_value=12 RegScheduler1.max_value=0.5 Pacevalue.min_value=0.2 Pacevalue.max_value=8 Trainer.checkpoint_path=runs/${save_dir}/spleen_our01_run2 Trainer.save_dir=${save_dir}/spleen_our01_run2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py num_models=4 seed=231 usetransform=True Dataset=spleen Trainer.name=CoMTSelf Arch.num_classes=2 Optim.lr=0.000003 Data.seed=12 AlphaScheduler.max_value=0.000015 RegScheduler.max_value=12 RegScheduler1.max_value=0.5 Pacevalue.min_value=0.2 Pacevalue.max_value=8 Trainer.checkpoint_path=runs/${save_dir}/spleen_our01_run3 Trainer.save_dir=${save_dir}/spleen_our01_run3 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done

