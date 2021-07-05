#!/usr/bin/env bash

save_dir=acdc_SP_our_0.05

num_batches=200
ratio1=0.05
unlab_ratio1=$(python -c "print(1-${ratio1})")

declare -a StringArray=(

"python -O main.py Optim.lr=0.00001 Trainer.name=CoMTSelf num_models=4 AlphaScheduler.max_value=0.00001 lam=0.02 RegScheduler.max_value=4 RegScheduler1.max_value=0.5 Trainer.save_dir=${save_dir}/0.00001our_0.02 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Optim.lr=0.00001 Trainer.name=CoMTSelf num_models=4 AlphaScheduler.max_value=0.0001 lam=0.1 RegScheduler.max_value=4 RegScheduler1.max_value=0.5 Trainer.save_dir=${save_dir}/0.0001our_0.1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Optim.lr=0.00001 Trainer.name=CoMTSelf num_models=4 AlphaScheduler.max_value=0.0001 lam=0.2 RegScheduler.max_value=4 RegScheduler1.max_value=0.5 Trainer.save_dir=${save_dir}/0.0001our_0.2 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

"python -O main.py Optim.lr=0.00001 Trainer.name=CoMTSelf num_models=4 AlphaScheduler.max_value=0.00005 lam=0.02 RegScheduler.max_value=4 RegScheduler1.max_value=0.5 Trainer.save_dir=${save_dir}/0.00005our_0.1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Optim.lr=0.00001 Trainer.name=CoMTSelf num_models=4 AlphaScheduler.max_value=0.00005 lam=0.05 RegScheduler.max_value=4 RegScheduler1.max_value=0.5 Trainer.save_dir=${save_dir}/0.00005our_0.1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Optim.lr=0.00001 Trainer.name=CoMTSelf num_models=4 AlphaScheduler.max_value=0.00005 lam=0.1 RegScheduler.max_value=4 RegScheduler1.max_value=0.5 Trainer.save_dir=${save_dir}/0.00005our_0.1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Optim.lr=0.00001 Trainer.name=CoMTSelf num_models=4 AlphaScheduler.max_value=0.00005 lam=0.2 RegScheduler.max_value=4 RegScheduler1.max_value=0.5 Trainer.save_dir=${save_dir}/0.00005our_0.1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

)

gpuqueue "${StringArray[@]}" --available_gpus 0 1

