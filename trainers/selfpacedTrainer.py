import torch
from deepclustering.utils import class2one_hot

from LOSS_JSD.functool import computeWeight
from LOSS_JSD.helper import average_list
from LOSS_JSD.jsd import GJSD_div
from trainers.Trainer import Trainer


class SelfPaceTrainer(Trainer):

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
        self._gjsd_criterion = GJSD_div()

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

        ensemble = torch.zeros_like(unlab_preds[0])
        for num in range(num_models):
            ensemble = ensemble + unlab_preds[num]
        ensemble = ensemble / num_models

        psueduo = torch.where(ensemble > 0.8, torch.tensor([1], dtype=torch.float).to(self._device), torch.tensor([0], dtype=torch.float).to(self._device))
        wk = computeWeight(preds=unlab_preds, psueduo=psueduo, labm=self._pace_scheduler.value)

        jsd_term1, jsd_term2 = self._gjsd_criterion(unlab_preds, wk)
        jsd_term11 = jsd_term1.item()
        jsd_term22 = jsd_term2.item()
        reg_jsdloss = jsd_term1 - (1 - self._alpha_scheduler.value) * jsd_term2

        for i in range(num_models):
            self._meter_interface[f"tra{i}_dice"].add(
                lab_preds[i].max(1)[1],
                target.squeeze(1),
                group_name=["_".join(x.split("_")[:-2]) for x in filename],
            )
        if self.train_jsd:
            self._meter_interface["jsd_term1"].add(jsd_term11)
            self._meter_interface["jsd_term2"].add(jsd_term22)
        return loss, reg_loss, reg_jsdloss



