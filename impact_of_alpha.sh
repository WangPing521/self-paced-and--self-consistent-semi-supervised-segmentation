#!/usr/bin/env bash
save_dir=impact_loss_curve80

num_batches=200
ratio1=0.1
unlab_ratio1=$(python -c "print(1-${ratio1})")

declare -a StringArray=(

"python -O main.py Arch.num_classes=4 Optim.lr=0.0001 Trainer.max_epoch=80 usetransform=True StartTraining.train_jsd=True num_models=2 RegScheduler.max_value=0.1 AlphaScheduler.begin_epoch=100 AlphaScheduler.max_value=0.0 Trainer.checkpoint_path=runs/${save_dir}/a_co_training Trainer.save_dir=${save_dir}/a_co_training Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Arch.num_classes=4 Optim.lr=0.0001 Trainer.max_epoch=80 usetransform=True StartTraining.train_jsd=True num_models=2 RegScheduler.max_value=0.1 AlphaScheduler.begin_epoch=100 AlphaScheduler.max_value=0.0 StartTraining.entropy_min=True RegScheduler1.begin_epoch=0 RegScheduler1.max_epoch=0 RegScheduler1.max_value=0.001 Trainer.checkpoint_path=runs/${save_dir}/a_co_training_0.001entropy Trainer.save_dir=${save_dir}/a_co_training_0.001entropy Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Arch.num_classes=4 Optim.lr=0.0001 Trainer.max_epoch=80 usetransform=True StartTraining.train_jsd=True num_models=2 RegScheduler.max_value=0.1 AlphaScheduler.begin_epoch=100 AlphaScheduler.max_value=0.0 StartTraining.entropy_min=True RegScheduler1.begin_epoch=0 RegScheduler1.max_epoch=0 RegScheduler1.max_value=0.0001 Trainer.checkpoint_path=runs/${save_dir}/a_co_training_0.0001entropy Trainer.save_dir=${save_dir}/a_co_training_0.0001entropy Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Arch.num_classes=4 Optim.lr=0.0001 Trainer.max_epoch=80 usetransform=True StartTraining.train_jsd=True num_models=2 RegScheduler.max_value=0.1 AlphaScheduler.begin_epoch=100 AlphaScheduler.max_value=0.0 StartTraining.entropy_min=True RegScheduler1.begin_epoch=0 RegScheduler1.max_epoch=0 RegScheduler1.max_value=0.00001 Trainer.checkpoint_path=runs/${save_dir}/a_co_training_0.00001entropy Trainer.save_dir=${save_dir}/a_co_training_0.00001entropy Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Arch.num_classes=4 Optim.lr=0.0001 Trainer.max_epoch=80 usetransform=True StartTraining.train_jsd=True num_models=2 RegScheduler.max_value=0.1 AlphaScheduler.begin_epoch=100 AlphaScheduler.max_value=0.0 StartTraining.entropy_min=True RegScheduler1.begin_epoch=0 RegScheduler1.max_epoch=0 RegScheduler1.max_value=0.000001 Trainer.checkpoint_path=runs/${save_dir}/a_co_training_0.000001entropy Trainer.save_dir=${save_dir}/a_co_training_0.000001entropy Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

"python -O main.py Arch.num_classes=2 Optim.lr=0.00001 Trainer.max_epoch=80 Dataset=prostate StartTraining.train_jsd=True num_models=2 RegScheduler.max_value=0.001 AlphaScheduler.begin_epoch=100 AlphaScheduler.max_value=0.0 Trainer.checkpoint_path=runs/${save_dir}/p_co_training Trainer.save_dir=${save_dir}/p_co_training Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Arch.num_classes=2 Optim.lr=0.00001 Trainer.max_epoch=80 Dataset=prostate StartTraining.train_jsd=True num_models=2 RegScheduler.max_value=0.001 AlphaScheduler.begin_epoch=100 AlphaScheduler.max_value=0.0 StartTraining.entropy_min=True RegScheduler1.begin_epoch=0 RegScheduler1.max_epoch=0 RegScheduler1.max_value=0.001 Trainer.checkpoint_path=runs/${save_dir}/p_co_training_0.001entropy Trainer.save_dir=${save_dir}/p_co_training_0.001entropy Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Arch.num_classes=2 Optim.lr=0.00001 Trainer.max_epoch=80 Dataset=prostate StartTraining.train_jsd=True num_models=2 RegScheduler.max_value=0.001 AlphaScheduler.begin_epoch=100 AlphaScheduler.max_value=0.0 StartTraining.entropy_min=True RegScheduler1.begin_epoch=0 RegScheduler1.max_epoch=0 RegScheduler1.max_value=0.0001 Trainer.checkpoint_path=runs/${save_dir}/p_co_training_0.0001entropy Trainer.save_dir=${save_dir}/p_co_training_0.0001entropy Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Arch.num_classes=2 Optim.lr=0.00001 Trainer.max_epoch=80 Dataset=prostate StartTraining.train_jsd=True num_models=2 RegScheduler.max_value=0.001 AlphaScheduler.begin_epoch=100 AlphaScheduler.max_value=0.0 StartTraining.entropy_min=True RegScheduler1.begin_epoch=0 RegScheduler1.max_epoch=0 RegScheduler1.max_value=0.00001 Trainer.checkpoint_path=runs/${save_dir}/p_co_training_0.00001entropy Trainer.save_dir=${save_dir}/p_co_training_0.00001entropy Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Arch.num_classes=2 Optim.lr=0.00001 Trainer.max_epoch=80 Dataset=prostate StartTraining.train_jsd=True num_models=2 RegScheduler.max_value=0.001 AlphaScheduler.begin_epoch=100 AlphaScheduler.max_value=0.0 StartTraining.entropy_min=True RegScheduler1.begin_epoch=0 RegScheduler1.max_epoch=0 RegScheduler1.max_value=0.000001 Trainer.checkpoint_path=runs/${save_dir}/p_co_training_0.000001entropy Trainer.save_dir=${save_dir}/p_co_training_0.000001entropy Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

)
gpuqueue "${StringArray[@]}" --available_gpus 2 6






