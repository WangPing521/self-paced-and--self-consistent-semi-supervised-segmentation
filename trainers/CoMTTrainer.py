
from deepclustering.utils import class2one_hot

from LOSS_JSD.helper import average_list
from trainers.Trainer import Trainer
from transformation import TensorRandomFlip, transforms_for_rot, transforms_back_rot
from torch.nn import functional as F


class CoMTTrainer(Trainer):

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
        self._weight_scheduler1 = weight_scheduler1
        self._transformer = TensorRandomFlip(axis=[1, 2])

    def _run_step(self, lab_data, unlab_data, *args, **kwargs):
        num_models = self._config['num_models']
        image, target, filename = (
            lab_data[0][0].to(self._device),
            lab_data[0][1].to(self._device),
            lab_data[1],
        )
        uimage = unlab_data[0][0].to(self._device)
        lab_preds, unlab_predt, unlab_preds_sf, preds_S = [], [], [], []
        for i in range(int(num_models / 2)):
            lab_preds.append(self._model[i](image).softmax(1))
            unlab_predt.append(self._model[int(i + int(num_models / 2))](uimage).softmax(1))

        if self._config['usetransform']:
            uimage_sf, rot_mask = transforms_for_rot(uimage)
            for i in range(int(num_models/2)):
                preds_S.append(self._model[i](uimage_sf).softmax(1))
            for s_pre in range(len(preds_S)):
                unlab_preds_sf.append(transforms_back_rot(preds_S[s_pre], rot_mask))
        else:
            for i in range(int(num_models / 2)):
                unlab_preds_sf.append(self._model[i](uimage).softmax(1))

        loss, consistency_loss = 0.0, 0.0
        cons_loss_way = self._config['cons_term']

        consistency_loss_list = []
        if cons_loss_way == 'MSE':
            for i in range(len(unlab_preds_sf)):
                consistency_loss_list.append(F.mse_loss(unlab_preds_sf[i], unlab_predt[i]))
            consistency_loss = average_list(consistency_loss_list)
        elif cons_loss_way == 'KL':
            for i in range(len(unlab_preds_sf)):
                consistency_loss_list.append(self._KL_criterion(unlab_preds_sf[i], unlab_predt[i]))
            consistency_loss = average_list(consistency_loss_list)

        onehot_target = class2one_hot(
            target.squeeze(1), self._model[0]._torchnet.num_classes
        )

        sup_loss_way = self._config['supvised_term']

        if sup_loss_way == 'cross_entropy':
            sup_loss = []
            for i in range(int(num_models/2)):
                sup_loss.append(self._ce_criterion(lab_preds[i], onehot_target))
            loss = average_list(sup_loss)
        elif sup_loss_way == 'dice_loss':
            sup_loss = []
            for i in range(int(num_models/2)):
                sup_loss.append(self._dice_loss(lab_preds[i], onehot_target))
            loss = average_list(sup_loss).mean()
        elif sup_loss_way == 'kl_loss':
            sup_loss = []
            for i in range(int(num_models/2)):
                sup_loss.append(self._KL_criterion(lab_preds[i], onehot_target))
            loss = average_list(sup_loss).mean()

        for i in range(int(num_models/2)):
            self._meter_interface[f"tra{i}_dice"].add(
                lab_preds[i].max(1)[1],
                target.squeeze(1),
                group_name=["_".join(x.split("_")[:-2]) for x in filename],
            )

        jsd_term1, jsd_term2 = self._jsd_criterion(unlab_preds_sf)
        reg_jsdloss = jsd_term1 - (1 - self._alpha_scheduler.value) * jsd_term2

        return loss, consistency_loss, reg_jsdloss



