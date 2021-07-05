#!/usr/bin/env bash

save_dir=prostate_our

num_batches=200
ratio1=0.1
unlab_ratio1=$(python -c "print(1-${ratio1})")

ratio2=0.05
unlab_ratio2=$(python -c "print(1-${ratio2})")

ratio3=0.07
unlab_ratio3=$(python -c "print(1-${ratio3})")

declare -a StringArray=(

"python -O main.py num_models=4 usetransform=True Dataset=prostate Trainer.name=CoMTSelf Arch.num_classes=2 Optim.lr=0.00001 AlphaScheduler.max_value=0.2 RegScheduler.max_value=4 RegScheduler1.max_value=0.1 Pacevalue.min_value=0.4 Pacevalue.max_value=4 Trainer.checkpoint_path=runs/${save_dir}/our_044_run1 Trainer.save_dir=${save_dir}/prostate/${ratio1}/our_044_run1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py num_models=4 usetransform=True seed=213 Dataset=prostate Trainer.name=CoMTSelf Arch.num_classes=2 Optim.lr=0.00001 AlphaScheduler.max_value=0.2 RegScheduler.max_value=4 RegScheduler1.max_value=0.1 Pacevalue.min_value=0.4 Pacevalue.max_value=4 Trainer.checkpoint_path=runs/${save_dir}/our_044_run2 Trainer.save_dir=${save_dir}/our_044_run2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py num_models=4 usetransform=True seed=231 Dataset=prostate Trainer.name=CoMTSelf Arch.num_classes=2 Optim.lr=0.00001 AlphaScheduler.max_value=0.2 RegScheduler.max_value=4 RegScheduler1.max_value=0.1 Pacevalue.min_value=0.4 Pacevalue.max_value=4 Trainer.checkpoint_path=runs/${save_dir}/our_044_run3 Trainer.save_dir=${save_dir}/our_044_run3 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"


)
gpuqueue "${StringArray[@]}" --available_gpus 2 6