import os

from deepclustering.trainer import _Trainer

from deepclustering import ModelMode
from deepclustering.meters import AverageValueMeter
from typing import Union
import torch
from deepclustering2.decorator.decorator import _disable_tracking_bn_stats
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter
from deepclustering.loss import KL_div, Entropy
from LOSS_JSD.helper import ModelList, average_list
from LOSS_JSD.jsd import JSD_div, SimplexCrossEntropyLoss
from LOSS_JSD.dice_loss import DiceLoss
from deepclustering.schedulers import Weight_RampScheduler
from deepclustering.model import ZeroGradientBackwardStep
from deepclustering.utils import (
    tqdm_,
    class2one_hot,
    flatten_dict,
    nice_dict,
    Path,
)

from record_meters.general_dice_meter import UniversalDice
from record_meters.save_images import save_images


class Trainer(_Trainer):
    this_directory = os.path.abspath(os.path.dirname(__file__))
    PROJECT_PATH = os.path.dirname(this_directory)
    RUN_PATH = str(Path(PROJECT_PATH, "runs"))

    def __init__(
            self,
            model: ModelList,
            lab_loader: Union[DataLoader, _BaseDataLoaderIter],
            unlab_loader: Union[DataLoader, _BaseDataLoaderIter],
            val_loader: DataLoader,
            weight_scheduler: Weight_RampScheduler = None,
            weight_scheduler1: Weight_RampScheduler = None,
            alpha_scheduler: Weight_RampScheduler = None,
            pace_scheduler: Weight_RampScheduler = None,
            max_epoch: int = 100,
            save_dir: str = "base",
            checkpoint_path: str = None,
            device="cpu",
            config: dict = None,
            num_batches=100,
            *args,
            **kwargs,
    ) -> None:
        self._lab_loader = lab_loader
        self._unlab_loader = unlab_loader
        super().__init__(
            model,
            None,
            val_loader,
            max_epoch,
            save_dir,
            checkpoint_path,
            device,
            config,
            *args,
            **kwargs,
        )
        self.smooth = config['Dice_config']['smooth']
        self.p = config['Dice_config']['p']
        self._num_batches = num_batches
        self._ce_criterion = SimplexCrossEntropyLoss()
        self._KL_criterion = KL_div()
        self._dice_loss = DiceLoss(smooth=self.smooth, p=self.p)
        self._entropy_criterion = Entropy()
        self._jsd_criterion = JSD_div()
        self._weight_scheduler = weight_scheduler
        self._weight_scheduler1 = weight_scheduler1
        self._alpha_scheduler = alpha_scheduler
        self._pace_scheduler = pace_scheduler
        self.checkpoint_path = checkpoint_path

    def register_meters(self, enable_drawer=True) -> None:
        super(Trainer, self).register_meters()
        c = self._config['Arch'].get('num_classes')
        num_models = self._config['num_models']
        report_axises = []
        for axi in range(c):
            report_axises.append(axi)
        if self._config['Trainer']['name'] == "meanteacher" or self._config['Trainer']['name'] == "uncertaintyMT" or \
                self._config['Trainer']['name'] == "co_mt":
            for i in range(int(num_models / 2)):
                self._meter_interface.register_new_meter(
                    f"tra{i}_dice", UniversalDice(C=c, report_axises=report_axises), group_name="train"
                )
        else:
            for i in range(num_models):
                self._meter_interface.register_new_meter(
                    f"tra{i}_dice", UniversalDice(C=c, report_axises=report_axises), group_name="train"
                )
        for i in range(num_models):
            self._meter_interface.register_new_meter(
                f"val{i}_dice", UniversalDice(C=c, report_axises=report_axises), group_name="val"
            )

        self._meter_interface.register_new_meter(
            f"val_ensemble_dice", UniversalDice(C=c, report_axises=report_axises), group_name="val"
        )
        self._meter_interface.register_new_meter(
            f"val_ensemble_wholedice", UniversalDice(C=c, report_axises=[0, 1]), group_name="val"
        )

        self._meter_interface.register_new_meter(
            "sup_loss", AverageValueMeter(), group_name="train"
        )
        self._meter_interface.register_new_meter(
            "entropy_loss", AverageValueMeter(), group_name="train"
        )
        self._meter_interface.register_new_meter(
            "jsd_ent_loss", AverageValueMeter(), group_name="train"
        )
        self._meter_interface.register_new_meter(
            "jsd_term1", AverageValueMeter(), group_name="train"
        )
        self._meter_interface.register_new_meter(
            "jsd_term2", AverageValueMeter(), group_name="train"
        )
        self._meter_interface.register_new_meter(
            "reg_jsdloss", AverageValueMeter(), group_name="train"
        )
        self._meter_interface.register_new_meter(
            "total_loss", AverageValueMeter(), group_name="train"
        )
        self._meter_interface.register_new_meter(
            "cons_loss", AverageValueMeter(), group_name="train"
        )
        self._meter_interface.register_new_meter(
            "co_weight", AverageValueMeter(), group_name="train"
        )
        self._meter_interface.register_new_meter(
            "threshhold", AverageValueMeter(), group_name="train"
        )
        self._meter_interface.register_new_meter(
            "alpha_value", AverageValueMeter(), group_name="train"
        )

    def _train_loop(
            self,
            lab_loader: Union[DataLoader, _BaseDataLoaderIter] = None,
            unlab_loader: Union[DataLoader, _BaseDataLoaderIter] = None,
            epoch: int = 0,
            mode=ModelMode.TRAIN,
            *args,
            **kwargs,
    ):
        self._model.set_mode(mode)
        s_co = min(1 - 1 / (epoch + 1), 0.999)
        batch_indicator = tqdm_(range(self._num_batches))
        batch_indicator.set_description(f"Training Epoch {epoch:03d}")

        for batch_id, lab_data, unlab_data in zip(
                batch_indicator, lab_loader, unlab_loader
        ):
            if self._config['Trainer']['name'] == "meanteacher" or self._config['Trainer']['name'] == "uncertaintyMT":
                loss, consistency_loss = self.run_step(lab_data=lab_data, unlab_data=unlab_data)
                with ZeroGradientBackwardStep(
                        loss + self._weight_scheduler.value * consistency_loss,
                        self._model[0]
                ) as new_loss:
                    new_loss.backward()
                for ema_param, param in zip(self._model[1].parameters(), self._model[0].parameters()):
                    ema_param.data.mul_(s_co).add_(1 - s_co, param.data)

                self._meter_interface["cons_loss"].add(consistency_loss.item())
            elif self._config['Trainer']['name'] == "co_mt" or self._config['Trainer']['name'] == "CoMTSelf":
                loss, consistency_loss, reg = self.run_step(lab_data=lab_data, unlab_data=unlab_data)
                if len(self._model) == 4:
                    with ZeroGradientBackwardStep(
                            loss + self._weight_scheduler.value * consistency_loss + self._weight_scheduler1.value * reg,
                            ModelList([self._model[0], self._model[1]]),
                    ) as new_loss:
                        new_loss.backward()
                elif len(self._model) == 6:
                    with ZeroGradientBackwardStep(
                            loss + self._weight_scheduler.value * consistency_loss + self._weight_scheduler1.value * reg,
                            ModelList([self._model[0], self._model[1], self._model[2]]),
                    ) as new_loss:
                        new_loss.backward()
                for num_m in range(int(len(self._model) / 2)):
                    for ema_param, param in zip(self._model[num_m + int(len(self._model) / 2)].parameters(),
                                                self._model[num_m].parameters()):
                        ema_param.data.mul_(s_co).add_(1 - s_co, param.data)
            else:
                loss, reg_loss, reg_jsdloss = self.run_step(lab_data=lab_data, unlab_data=unlab_data)
                with ZeroGradientBackwardStep(
                        loss + self._weight_scheduler.value * reg_jsdloss + self._weight_scheduler1.value * reg_loss,
                        self._model
                ) as new_loss:
                    new_loss.backward()

            self._meter_interface["total_loss"].add(new_loss.item())
            self._meter_interface["sup_loss"].add(loss.item())
            if self.entropy_min and self.train_jsd:
                self._meter_interface["jsd_ent_loss"].add((reg_loss + reg_jsdloss).item())
            if self.entropy_min:
                self._meter_interface["entropy_loss"].add(reg_loss.item())
            if self.train_jsd or self._config['Trainer']['name'] == "selfpace":
                self._meter_interface["reg_jsdloss"].add(reg_jsdloss.item())

            if ((batch_id + 1) % 5) == 0:
                report_statue = self._meter_interface.tracking_status("train")
                batch_indicator.set_postfix(flatten_dict(report_statue))
        report_statue = self._meter_interface.tracking_status("train")
        batch_indicator.set_postfix(flatten_dict(report_statue))
        self.writer.add_scalar_with_tag(
            "train", flatten_dict(report_statue), global_step=epoch
        )
        print(f"Training Epoch {epoch}: {nice_dict(flatten_dict(report_statue))}")

    def _eval_loop(
            self,
            val_loader: Union[DataLoader, _BaseDataLoaderIter] = None,
            epoch: int = 0,
            mode=ModelMode.EVAL,
            *args,
            **kwargs,
    ) -> float:
        self._model.set_mode(mode)
        val_indicator = tqdm_(val_loader)
        val_indicator.set_description(f"Validation Epoch {epoch:03d}")
        num_models = self._config['num_models']
        for batch_id, data in enumerate(val_indicator):
            image, target, filename = (
                data[0][0].to(self._device),
                data[0][1].to(self._device),
                data[1],
            )
            ensembel_pred = []
            # with _disable_tracking_bn_stats(self._model[1]):
            for i in range(num_models):
                preds = self._model[i](image).softmax(1)
                ensembel_pred.append(preds)
                self._meter_interface[f"val{i}_dice"].add(
                    preds.max(1)[1],
                    target.squeeze(1),
                    group_name=["_".join(x.split("_")[:2]) for x in filename])

            ensemble = torch.zeros_like(ensembel_pred[0])
            if self._config['Trainer']['name'] == "co_mt" or self._config['Trainer']['name'] == "CoMTSelf":
                for num in range(int(num_models / 2)):
                    ensemble = ensemble + ensembel_pred[num + int(num_models / 2)]
                ensemble = ensemble / int(num_models / 2)
            elif self._config['Trainer']['name'] == "meanteacher" or self._config['Trainer']['name'] == "uncertaintyMT":
                ensemble = ensemble + ensembel_pred[1]
            else:
                for num in range(num_models):
                    ensemble = ensemble + ensembel_pred[num]
                ensemble = ensemble / num_models

            self._meter_interface[f"val_ensemble_dice"].add(
                ensemble.max(1)[1],
                target.squeeze(1),
                group_name=["_".join(x.split("_")[:2]) for x in filename])

            whole_ensemble = ensemble.max(1)[1]
            whole_ensemble = torch.where(whole_ensemble > 0, torch.tensor([1]).to(self._device),
                                         torch.tensor([0]).to(self._device))
            whole_target = torch.where(target.squeeze(1) > 0, torch.tensor([1]).to(self._device),
                                       torch.tensor([0]).to(self._device))
            self._meter_interface[f"val_ensemble_wholedice"].add(
                whole_ensemble,
                whole_target,
                group_name=["_".join(x.split("_")[:2]) for x in filename])
            # entropy_val = -ensemble * torch.log(ensemble + 0.000000000000000000001)
            # save_images(entropy_val[:,1,:,:], names=filename, root=self._config['Trainer']['save_dir'], mode='entropy_rv', iter=epoch)
            if epoch == 99:
                save_images(ensemble.max(1)[1], names=filename, root=self._config['Trainer']['save_dir'], mode='prediction', iter=epoch)
            # save_images(ensembel_pred[0].max(1)[1], names=filename, root=self._config['Trainer']['save_dir'], mode='prediction1', iter=epoch)
            # save_images(ensembel_pred[1].max(1)[1], names=filename, root=self._config['Trainer']['save_dir'], mode='prediction2', iter=epoch)

            if ((batch_id + 1) % 5) == 0:
                report_statue = self._meter_interface.tracking_status("val")
                val_indicator.set_postfix(flatten_dict(report_statue))
        report_statue = self._meter_interface.tracking_status("val")
        val_indicator.set_postfix(flatten_dict(report_statue))
        self.writer.add_scalar_with_tag(
            "val", flatten_dict(report_statue), global_step=epoch
        )
        print(f"Validation Epoch {epoch}: {nice_dict(flatten_dict(report_statue))}")

        return average_list(
            [
                average_list(self._meter_interface[f"val{n}_dice"].summary().values())
                for n in range(num_models)
            ]
        )

    def schedulerStep(self):

        for segmentator in self._model:
            segmentator.schedulerStep()
        self._weight_scheduler.step()
        self._weight_scheduler1.step()
        self._alpha_scheduler.step()
        self._pace_scheduler.step()

    def _start_training(self):
        self.to(self._device)
        entropy_min = self._config['StartTraining'].get('entropy_min')
        train_jsd = self._config['StartTraining'].get('train_jsd')
        self.entropy_min = entropy_min
        self.train_jsd = train_jsd
        for epoch in range(self._start_epoch, self._max_epoch):
            if self._model.get_lr() is not None:
                self._meter_interface["lr"].add(self._model.get_lr()[0])

            self._meter_interface["co_weight"].add(self._weight_scheduler.value)
            self._meter_interface["threshhold"].add(self._weight_scheduler1.value)
            if self.train_jsd:
                self._meter_interface["alpha_value"].add(self._alpha_scheduler.value)
            self.train_loop(
                lab_loader=self._lab_loader,
                unlab_loader=self._unlab_loader,
                epoch=epoch
            )
            with torch.no_grad():
                current_score = self.eval_loop(self._val_loader, epoch)
            self.schedulerStep()
            # save meters and checkpoints
            self._meter_interface.step()
            if hasattr(self, "_dataframe_drawer"):
                self._dataframe_drawer()
            self.save_checkpoint(self.state_dict(), epoch, current_score)
            self._meter_interface.summary().to_csv(self._save_dir / "wholeMeter.csv")


class TwoStageTrainer(Trainer):
    def __init__(self, model,
                 lab_loader,
                 unlab_loader,
                 val_loader,
                 weight_scheduler,
                 weight_scheduler1,
                 alpha_scheduler,
                 pace_scheduler,
                 max_epoch,
                 save_dir,
                 checkpoint_path: str = None,
                 device="cpu",
                 config: dict = None,
                 num_batches=100,
                 *args,
                 **kwargs):
        Trainer.__init__(self, model,
                         lab_loader,
                         unlab_loader,
                         val_loader,
                         weight_scheduler,
                         weight_scheduler1,
                         alpha_scheduler,
                         pace_scheduler,
                         max_epoch,
                         save_dir,
                         checkpoint_path,
                         device,
                         config,
                         num_batches,
                         *args,
                         **kwargs)

    def _run_step(self, lab_data, unlab_data, *args, **kwargs):
        num_models = self._config['num_models']
        image, target, filename = (
            lab_data[0][0].to(self._device),
            lab_data[0][1].to(self._device),
            lab_data[1],
        )
        uimage = unlab_data[0][0].to(self._device)
        lab_preds, unlab_preds = [], []
        for i in range(num_models):
            lab_preds.append(self._model[i](image).softmax(1))
            unlab_preds.append(self._model[i](uimage).softmax(1))

        onehot_target = class2one_hot(
            target.squeeze(1), self._model[0]._torchnet.num_classes
        )

        loss, reg_loss, jsd_term1, jsd_term2, reg_jsdloss = 0.0, 0.0, 0.0, 0.0, 0.0

        sup_loss_way = self._config['supvised_term']
        if sup_loss_way == 'cross_entropy':
            sup_loss = []
            for i in range(num_models):
                sup_loss.append(self._ce_criterion(lab_preds[i], onehot_target))
            loss = average_list(sup_loss)
        elif sup_loss_way == 'dice_loss':
            sup_loss = []
            for i in range(num_models):
                sup_loss.append(self._dice_loss(lab_preds[i], onehot_target))
            loss = average_list(sup_loss).mean()
        elif sup_loss_way == 'kl_loss':
            sup_loss = []
            for i in range(num_models):
                sup_loss.append(self._KL_criterion(lab_preds[i], onehot_target))
            loss = average_list(sup_loss).mean()

        if self.entropy_min:
            reg_loss_list = []
            for i in range(num_models):
                reg_loss_list.append(self._entropy_criterion(unlab_preds[i]))
                reg_loss = average_list(reg_loss_list)
        if self.train_jsd:
            jsd_term1, jsd_term2 = self._jsd_criterion(unlab_preds)
            jsd_term11 = jsd_term1.item()
            jsd_term22 = jsd_term2.item()
            reg_jsdloss = jsd_term1 - (1 - self._alpha_scheduler.value) * jsd_term2
        for i in range(num_models):
            self._meter_interface[f"tra{i}_dice"].add(
                lab_preds[i].max(1)[1],
                target.squeeze(1),
                group_name=["_".join(x.split("_")[:2]) for x in filename],
            )
        if self.train_jsd:
            self._meter_interface["jsd_term1"].add(jsd_term11)
            self._meter_interface["jsd_term2"].add(jsd_term22)
        return loss, reg_loss, reg_jsdloss
