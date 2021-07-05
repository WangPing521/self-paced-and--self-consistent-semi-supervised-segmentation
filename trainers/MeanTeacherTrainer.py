from deepclustering.utils import class2one_hot
import torch
from trainers.Trainer import Trainer
from transformation import TensorRandomFlip, transforms_for_rot, transforms_back_rot
from torch.nn import functional as F


class MeanTeacherTrainer(Trainer):

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
        self._transformer = TensorRandomFlip(axis=[1, 2])

    def _run_step(self, lab_data, unlab_data, *args, **kwargs):

        image, target, filename = (
            lab_data[0][0].to(self._device),
            lab_data[0][1].to(self._device),
            lab_data[1],
        )
        uimage = unlab_data[0][0].to(self._device)
        lab_preds = self._model[0](image).softmax(1)
        unlab_predT = self._model[1](uimage).softmax(1)

        loss, consistency_loss = 0.0, 0.0
        cons_loss_way = self._config['cons_term']

        if self._config['usetransform']:
            uimage_sf, rot_mask = transforms_for_rot(uimage)
            unlab_preds_sf = self._model[0](uimage_sf).softmax(1)
            unlab_preds_stf = transforms_back_rot(unlab_preds_sf, rot_mask)
        else:
            unlab_preds_stf = self._model[0](uimage).softmax(1)
        if cons_loss_way == 'MSE':
            consistency_loss = F.mse_loss(unlab_preds_stf, unlab_predT)
        elif cons_loss_way == 'KL':
            consistency_loss = self._KL_criterion(unlab_preds_stf, unlab_predT)

        onehot_target = class2one_hot(
            target.squeeze(1), self._model[0]._torchnet.num_classes
        )

        sup_loss_way = self._config['supvised_term']

        if sup_loss_way == 'cross_entropy':
            loss = self._ce_criterion(lab_preds, onehot_target)
        elif sup_loss_way == 'dice_loss':
            loss = self._dice_loss(lab_preds, onehot_target)
        elif sup_loss_way == 'kl_loss':
            loss = self._KL_criterion(lab_preds, onehot_target)

        self._meter_interface[f"tra{0}_dice"].add(
            lab_preds.max(1)[1],
            target.squeeze(1),
            group_name=["_".join(x.split("_")[:-2]) for x in filename],
        )

        return loss, consistency_loss