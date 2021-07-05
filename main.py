import sys

sys.path.insert(0, "../")
from LOSS_JSD.helper import ModelList
from deepclustering import __git_hash__ as dc_hash
from augment import train_transform, val_transform
from augment_spleen import train_transformS, val_transformS
from trainers.Trainer import TwoStageTrainer
from trainers.MeanTeacherTrainer import MeanTeacherTrainer
from trainers.uncertaintyMTTrainer import UncertaintyMTTrainer
from trainers.selfpacedTrainer import SelfPaceTrainer
from trainers.CoMTSelfPace import CoMTSelf
from trainers.CoMTTrainer import CoMTTrainer
from deepclustering.schedulers import Weight_RampScheduler
from scheduler.alphaScheduler import Alpha_RampScheduler
from deepclustering.dataset.segmentation import ACDCSemiInterface, SpleenSemiInterface, ProstateSemiInterface
from deepclustering.manager import ConfigManger
from deepclustering.model import Model
from deepclustering.utils import fix_all_seed


def adding_hash(config, **kwargs):
    return {**config, **kwargs}

config = ConfigManger("config_file/config.yaml").config
config = adding_hash(config, deepclustering_hash=dc_hash, multimodal_hash=hash)
fix_all_seed(config['seed'])
num_models = config['num_models']

if num_models == 1:
    model1 = Model(config["Arch"], config["Optim"], config["Scheduler"])
    models = ModelList([model1])
elif num_models == 2:
    model1 = Model(config["Arch"], config["Optim"], config["Scheduler"])
    model2 = Model(config["Arch"], config["Optim"], config["Scheduler"])
    if config['Trainer']['name'] == "meanteacher" or config['Trainer']['name'] == "uncertaintyMT":
        for param in model2.parameters():
            param.detach_()
    models = ModelList([model1, model2])
elif num_models == 3:
    model1 = Model(config["Arch"], config["Optim"], config["Scheduler"])
    model2 = Model(config["Arch"], config["Optim"], config["Scheduler"])
    model3 = Model(config["Arch"], config["Optim"], config["Scheduler"])
    models = ModelList([model1, model2, model3])
elif num_models == 4:
    model1 = Model(config["Arch"], config["Optim"], config["Scheduler"])
    model2 = Model(config["Arch"], config["Optim"], config["Scheduler"])
    model3 = Model(config["Arch"], config["Optim"], config["Scheduler"])
    model4 = Model(config["Arch"], config["Optim"], config["Scheduler"])
    for param in model3.parameters():
        param.detach_()
    for param in model4.parameters():
        param.detach_()
    models = ModelList([model1, model2, model3, model4])
elif num_models == 6:
    model1 = Model(config["Arch"], config["Optim"], config["Scheduler"])
    model2 = Model(config["Arch"], config["Optim"], config["Scheduler"])
    model3 = Model(config["Arch"], config["Optim"], config["Scheduler"])
    model4 = Model(config["Arch"], config["Optim"], config["Scheduler"])
    model5 = Model(config["Arch"], config["Optim"], config["Scheduler"])
    model6 = Model(config["Arch"], config["Optim"], config["Scheduler"])
    for param in model4.parameters():
        param.detach_()
    for param in model5.parameters():
        param.detach_()
    for param in model6.parameters():
        param.detach_()
    models = ModelList([model1, model2, model3, model4, model5, model6])

if config['Dataset'] == 'acdc':
    dataset_handler = ACDCSemiInterface(**config["Data"])
elif config['Dataset'] == 'spleen':
    dataset_handler = SpleenSemiInterface(**config["Data"])
    train_transform = train_transformS
    val_transform = val_transformS
elif config['Dataset'] == 'prostate':
    dataset_handler = ProstateSemiInterface(**config["Data"])


def get_group_set(dataloader):
    return set(sorted(dataloader.dataset.get_group_list()))


dataset_handler.compile_dataloader_params(**config["DataLoader"])
label_loader, unlab_loader, val_loader = dataset_handler.SemiSupervisedDataLoaders(
    labeled_transform=train_transform,
    unlabeled_transform=train_transform,
    val_transform=val_transform,
    group_val=True,
    use_infinite_sampler=True,
)
assert get_group_set(label_loader) & get_group_set(unlab_loader) == set()
assert (get_group_set(label_loader) | get_group_set(unlab_loader)) & get_group_set(val_loader) == set()
print(
    f"Labeled loader with {len(get_group_set(label_loader))} groups: \n {', '.join(sorted(get_group_set(label_loader))[:5])}"
)
print(
    f"Unabeled loader with {len(get_group_set(unlab_loader))} groups: \n {', '.join(sorted(get_group_set(unlab_loader))[:5])}"
)
print(
    f"Val loader with {len(get_group_set(val_loader))} groups: \n {', '.join(sorted(get_group_set(val_loader))[:5])}"
)


RegScheduler = Weight_RampScheduler(**config["RegScheduler"])
RegScheduler1 = Weight_RampScheduler(**config["RegScheduler1"])
AlphaScheduler = Alpha_RampScheduler(**config["AlphaScheduler"])
PaceScheduler = Weight_RampScheduler(**config["Pacevalue"])


TrainerT = {
    "twostage": TwoStageTrainer,
    "meanteacher": MeanTeacherTrainer,
    "co_mt": CoMTTrainer,
    "selfpace": SelfPaceTrainer,
    "CoMTSelf": CoMTSelf,
    "uncertaintyMT": UncertaintyMTTrainer
}

Trainer_T = TrainerT.get(config['Trainer'].get('name'))

trainer = Trainer_T(
    model=models,
    lab_loader=label_loader,
    unlab_loader=unlab_loader,
    weight_scheduler=RegScheduler,
    weight_scheduler1=RegScheduler1,
    alpha_scheduler=AlphaScheduler,
    pace_scheduler=PaceScheduler,
    val_loader=val_loader,
    config=config,
    **config["Trainer"],
)
# trainer.inference(identifier="last.pth")
trainer.start_training()
