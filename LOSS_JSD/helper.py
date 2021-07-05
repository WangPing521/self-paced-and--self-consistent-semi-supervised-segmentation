from functools import reduce
from typing import Union

import numpy as np
import torch
from torch import Tensor, Size
from torch.utils.data.dataloader import _BaseDataLoaderIter, DataLoader

from deepclustering.dataloader.sampler import InfiniteRandomSampler
from deepclustering.utils import simplex


def average_list(input_list):
    return sum(input_list) / len(input_list)


class ModelList(list):
    """
    This is to help to save a list of models
    without changing or extending existing apis
    of `deep-clustering-toolbox`.
    """

    @property
    def use_apex(self):
        if hasattr(self[0], "_use_apex"):
            return self[0]._use_apex
        return False

    def parallel_call(self, *args, **kwargs):
        """unify the interface for both one model and multiple models."""
        result = []
        n = len(self)
        for i in range(n):
            result.append(self[i](*args, **kwargs))
        return result

    def serial_call(self, *args, **kwargs):
        assert len(args) == len(self), len(args)
        result = []
        n = len(self)
        for i in range(n):
            result.append(self[i](args[i], **kwargs))
        return result

    def state_dict(self):
        result_dict = {}
        n = len(self)
        for i in range(n):
            result_dict[i] = self[i].state_dict()
        return result_dict

    def load_state_dict(self, state_dict):
        n_input = len(state_dict.keys())
        n = len(self)
        assert n_input == n
        for i in range(n):
            self[i].load_state_dict(state_dict[i])

    def to(self, device):
        for i in range(len(self)):
            self[i].to(device)

    def schedulerStep(self):
        for i in range(len(self)):
            self[i].schedulerStep()

    def zero_grad(self):
        for i in range(len(self)):
            self[i].zero_grad()

    def step(self):
        for i in range(len(self)):
            self[i].step()

    def set_mode(self, mode):
        for i in range(len(self)):
            self[i].set_mode(mode)

    def get_lr(self):
        return self[0].get_lr()

    def apply(self, *args, **kwargs):
        for i in range(len(self)):
            self[i].apply(*args, **kwargs)

    def modules(self):
        for i in range(len(self)):
            yield from self[i]._torchnet.modules()


def unzip_2d_data(data, device):
    (modal1, modal2, target), filename = (
        (_data.to(device) for _data in data[0]),
        data[1],
    )
    return (modal1, modal2, target), filename


def unzip_3d_data(data, device):
    mixed_img, target, filename = data
    modal1, modal2 = mixed_img[:, 0].unsqueeze(1), mixed_img[:, 1].unsqueeze(1)
    modal1, modal2, target = modal1.to(device), modal2.to(device), target.to(device)
    return (modal1, modal2, target), list(filename)


def confident_mask_generator(preds: Tensor, threshold: float):
    assert simplex(preds), preds
    b, c, *hw = preds.shape
    assert 0 <= threshold <= 1, threshold
    mask = preds.max(1)[0] >= threshold
    assert mask.shape == Size([b, *hw])
    return mask.float()


def confident_mask_generator_from_simplex(
    preds: Tensor, current_epoch: int, max_epoch: int
):
    assert simplex(preds), preds
    percentile = float(current_epoch) / float(max_epoch)
    threshold = _confident_ranking(preds.max(1)[0], percentile * 100)
    return confident_mask_generator(preds, threshold)


def confident_mask_generator_from_confident_score(
    confident: Tensor, current_epoch: int, max_epoch: int
):
    assert simplex(preds), preds
    percentile = 1 - np.exp(-float(current_epoch) / float(max_epoch))
    threshold = _confident_ranking(confident, percentile)
    return (confident >= threshold).float().squeeze(1)


def _confident_ranking(preds, percentage):
    _preds = preds
    assert torch.all((_preds <= 1) & (_preds >= 0))
    threshold = __percentile(_preds, percentage)
    assert threshold >= 0 and threshold <= 1
    return threshold


def __percentile(t: torch.tensor, q: float) -> Union[int, float]:
    """
    Return the ``q``-th percentile of the flattened input tensor's data.

    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.

    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(0.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result


def _is_DataLoaderIter(dataloader) -> bool:
    """
    check if one dataset is DataIterator.
    :param dataloader:
    :return:
    """
    return isinstance(dataloader, _BaseDataLoaderIter)


def loader2Iter(dataloader) -> _BaseDataLoaderIter:
    if _is_DataLoaderIter(dataloader):
        return dataloader
    elif isinstance(dataloader, DataLoader):
        assert isinstance(
            dataloader.sampler, InfiniteRandomSampler  # type ignore
        ), "we hope the sampler should be InfiniteRanomSampler"
        return iter(dataloader)  # type ignore
    else:
        raise TypeError("given dataloader type of {}".format(type(dataloader)))


def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


group_set = lambda x: set(x.dataset.get_group_list())


def nointersect(*x):
    return reduce(lambda x, y: group_set(x) & group_set(y), x) == set()


def get_group(x):
    return x.dataset.get_group_list()


if __name__ == "__main__":
    preds = torch.randn(4, 4, 256, 256).softmax(1)
    confid_mask = confident_mask_generator_from_simplex(preds, 10, 100)

    confident = torch.randn(4, 256, 256).sigmoid()
    confi_mask2 = confident_mask_generator_from_confident_score(confident, 10, 100)
    pass
