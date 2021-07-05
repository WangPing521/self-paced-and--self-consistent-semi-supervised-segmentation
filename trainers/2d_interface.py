from itertools import repeat
from pathlib import Path
from typing import Callable, Tuple, List

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from deepclustering.meters import MeterInterface, AverageValueMeter
from deepclustering.utils import (
    iter_average,
    filter_dict,
    dict_flatten,
    pprint,
    tqdm_,
)
from deepclustering.writer import SummaryWriter
from multimodal.meters import UniversalDice, SurfaceMeter
from multimodal.utils import ModelList


def step_iter(modal0: torch.Tensor, modal1: torch.Tensor, batch_size=4):
    assert modal0.shape == modal1.shape
    modal0_batch, modal1_batch, batch_index = [], [], []
    for num, (m0, m1) in enumerate(zip(modal0, modal1)):
        modal0_batch.append(m0)
        modal1_batch.append(m1)
        batch_index.append(num)
        if len(modal0_batch) == batch_size:
            yield torch.stack(modal0_batch, dim=0), torch.stack(
                modal1_batch, dim=0
            ), batch_index
            modal0_batch, modal1_batch, batch_index = [], [], []
    yield torch.stack(modal0_batch, dim=0), torch.stack(
        modal1_batch, dim=0
    ), batch_index


class TrainerInferenceMixin:
    _model: ModelList
    _meter_interface: MeterInterface
    _val_loader: DataLoader
    writer: SummaryWriter
    predict: Callable[[ModelList, torch.Tensor, torch.Tensor], List[torch.Tensor]]
    num_models: int
    _device: torch.device
    _save_dir: Path
    unzip_data: Callable[
        [torch.Tensor, torch.device],
        Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], List[str]],
    ]
    process_group_name: Callable[[str], str]

    def _init_meters(self):
        self._meter_interface = MeterInterface()  # reset meter_interface

        for i in range(self.num_models):
            self._meter_interface.register_new_meter(
                f"model{i}_dice",
                UniversalDice(
                    C=self.num_class, report_axises=list(range(1, self.num_class))
                ),
                group_name="inference",
            )
            self._meter_interface.register_new_meter(
                f"model{i}_mhd",
                SurfaceMeter(
                    C=self.num_class,
                    report_axises=list(range(1, self.num_class)),
                    metername="mod_hausdorff",
                ),
                group_name="inference",
            )
            self._meter_interface.register_new_meter(
                f"model{i}_asd",
                SurfaceMeter(
                    C=self.num_class,
                    report_axises=list(range(1, self.num_class)),
                    metername="average_surface",
                ),
                group_name="inference",
            )
        self._meter_interface.register_new_meter(
            f"ensemble_dice",
            UniversalDice(
                C=self.num_class, report_axises=list(range(1, self.num_class))
            ),
            group_name="inference",
        )
        self._meter_interface.register_new_meter(
            "ensemble_dice_summary", AverageValueMeter(), group_name="inference"
        )
        self._meter_interface.register_new_meter(
            f"ensemble_mhd",
            SurfaceMeter(
                C=self.num_class,
                report_axises=list(range(1, self.num_class)),
                metername="mod_hausdorff",
            ),
            group_name="inference",
        )
        self._meter_interface.register_new_meter(
            "ensemble_mhd_summary", AverageValueMeter(), group_name="inference"
        )
        self._meter_interface.register_new_meter(
            f"ensemble_asd",
            SurfaceMeter(
                C=self.num_class,
                report_axises=list(range(1, self.num_class)),
                metername="average_surface",
            ),
            group_name="inference",
        )
        self._meter_interface.register_new_meter(
            "ensemble_asd_summary", AverageValueMeter(), "inference"
        )

    def _inference_predict(
        self, model: ModelList, modal0: torch.Tensor
    ) -> List[torch.Tensor]:
        assert modal0.device == torch.device("cpu")
        B, _, h, w = modal0.shape

        img_iter = step_iter(modal0, batch_size=10)

        whole_preds = [
            torch.zeros(
                B, self.num_class, h, w, dtype=torch.float, device=torch.device("cpu")
            )
            for _ in range(self.num_models)
        ]
        count_useds = [
            torch.zeros_like(
                whole_preds[0], dtype=torch.float, device=torch.device("cpu")
            )
            + 1e-10
            for _ in range(self.num_models)
        ]
        for (modal0_crop, batch_index) in tqdm_(img_iter):
            modal0_crop = modal0_crop.to(self._device)
            pred_crop: List[Tensor] = self.predict(model, modal0_crop)
            pred_crop = [x.to(torch.device("cpu")) for x in pred_crop]

            for _pred_crop, _count_used, whole_pred in zip(
                pred_crop, count_useds, whole_preds
            ):
                for i, b in enumerate(batch_index):
                    whole_pred[b] += _pred_crop[i].detach()
                    _count_used[b] += 1
        assert count_useds[0].min() >= 1, count_useds[0].min()
        whole_preds = [
            whole_pred / count_used
            for whole_pred, count_used in zip(whole_preds, count_useds)
        ]
        return whole_preds

    @staticmethod
    def _ensemble(whole_preds: List[torch.Tensor]) -> torch.Tensor:
        return iter_average(whole_preds)

    @torch.no_grad()
    def inference(
        self,
        identifier="last.pth",
        save_image=True,
        val_batch_loader=None,
        *args,
        **kwargs,
    ):
        assert isinstance(val_batch_loader, DataLoader), val_batch_loader
        print(f"inference function with identifier={identifier}.")
        super().inference(identifier=identifier, *args, **kwargs)  # load checkpoint.
        current_model = self._model
        if self._use_meanteacher:  # put teacher model to device.
            current_model = self._teacher_model
            self._teacher_model.to(self._device)

        self.num_class: int = self._model[0]._torchnet.num_classes

        self._init_meters()
        for index, _data in enumerate(val_batch_loader):
            (modal0, modal1, target), group_name = self.unzip_data(
                _data, device=torch.device("cpu")
            )
            whole_preds = self._inference_predict(current_model, modal0, modal1)
            if save_image:
                self.save_predictions(
                    whole_preds,
                    filenames=[
                        f"saved_images/{identifier}/{name}/model_{i}.pth"
                        for i, name in zip(
                            range(self.num_models), repeat(group_name[0])
                        )
                    ],
                )
                self.save_predictions(
                    [target],
                    filenames=[f"saved_images/{identifier}/{group_name[0]}/target.pth"],
                )
            ensemble_pred = self._ensemble(whole_preds)
            assert len(whole_preds) == self.num_models, len(whole_preds)
            for i in range(self.num_models):
                self._meter_interface[f"model{i}_dice"].add(
                    whole_preds[i].max(1)[1],
                    target.squeeze(1),
                    group_name=self.process_group_name(group_name),
                )
                self._meter_interface[f"model{i}_mhd"].add(
                    whole_preds[i].max(1)[1].unsqueeze(0),
                    target.squeeze(1).unsqueeze(0),
                )
                self._meter_interface[f"model{i}_asd"].add(
                    whole_preds[i].max(1)[1].unsqueeze(0),
                    target.squeeze(1).unsqueeze(0),
                )
            self._meter_interface["ensemble_dice"].add(
                ensemble_pred.max(1)[1],
                target.squeeze(1),
                group_name=self.process_group_name(group_name),
            )
            self._meter_interface["ensemble_mhd"].add(
                ensemble_pred.max(1)[1].unsqueeze(0), target.squeeze(1).unsqueeze(0),
            )
            self._meter_interface["ensemble_asd"].add(
                ensemble_pred.max(1)[1].unsqueeze(0), target.squeeze(1).unsqueeze(0),
            )
            report_dict = self._meter_interface.tracking_status(group_name="inference")

        self._meter_interface["ensemble_dice_summary"].add(
            iter_average(list(report_dict["ensemble_dice"].values()))
        )
        self._meter_interface["ensemble_mhd_summary"].add(
            iter_average(list(report_dict["ensemble_mhd"].values()))
        )
        self._meter_interface["ensemble_asd_summary"].add(
            iter_average(list(report_dict["ensemble_asd"].values()))
        )
        report_dict = self._meter_interface.tracking_status(group_name="inference")
        self.writer.add_scalar_with_tag(
            tag=f"inference_{identifier}",
            tag_scalar_dict=filter_dict(dict_flatten(report_dict)),
            global_step=1000,
        )

        self._meter_interface.step()
        self._meter_interface.summary().to_csv(
            self._save_dir / f"{identifier}_inference.csv"
        )
        pprint(report_dict)

    @staticmethod
    def _save_prediction(prediction, filename):
        filename.parent.mkdir(exist_ok=True, parents=True)
        torch.save(prediction.squeeze(0), f=filename)

    def save_predictions(self, predictions, filenames):
        save_dir = self._save_dir
        for filename, prediction in zip(filenames, predictions):
            self._save_prediction(prediction, save_dir / filename)
