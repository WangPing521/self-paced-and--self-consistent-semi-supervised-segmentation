from deepclustering.utils import class2one_hot
import torch
from deepclustering2.decorator.decorator import _disable_tracking_bn_stats

from LOSS_JSD.helper import average_list
from trainers.Trainer import Trainer
from transformation import TensorRandomFlip, transforms_for_rot, transforms_back_rot
from torch.nn import functional as F


class UncertaintyMTTrainer(Trainer):

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

        if self._config['usetransform']:
            uimage_sf, rot_mask = transforms_for_rot(uimage)
            unlab_preds_sf = self._model[0](uimage_sf).softmax(1)
            unlab_preds_stf = transforms_back_rot(unlab_preds_sf, rot_mask)
        else:
            unlab_preds_stf = self._model[0](uimage).softmax(1)

        consistency_loss_1 = F.mse_loss(unlab_preds_stf, unlab_predT, reduction='none')
        onehot_target = class2one_hot(
            target.squeeze(1), self._model[0]._torchnet.num_classes
        )
        loss = self._ce_criterion(lab_preds, onehot_target)

        # compute uncetainty
        _disable_tracking_bn_stats(self._model[1])
        T = 8
        unlab_predTgroup = []
        for t_num in range(T):
            noise = torch.clamp(torch.randn_like(uimage) * 0.1, -0.2, 0.2)
            uimage_noise = uimage + noise
            unlab_predTgroup.append(self._model[1](uimage_noise).softmax(1))
        _disable_tracking_bn_stats(self._model[1])
        unlab_avgpred = average_list(unlab_predTgroup)
        uncertainty = (- unlab_avgpred * torch.log(unlab_avgpred + 0.00000001)).sum(1).unsqueeze(1)
        certainty_mask = (uncertainty < self._weight_scheduler1.value).float()

        _, num_c, _, _ = unlab_predT.shape
        consistency_loss = torch.sum(consistency_loss_1 * certainty_mask)/(num_c*torch.sum(certainty_mask)+1e-16)

        self._meter_interface[f"tra{0}_dice"].add(
            lab_preds.max(1)[1],
            target.squeeze(1),
            group_name=["_".join(x.split("_")[:-2]) for x in filename],
        )

        return loss, consistency_loss