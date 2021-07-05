from typing import List, Optional

import torch
from deepclustering.loss import KL_div, Union, assert_list, simplex
from deepclustering.loss.kl_losses import _check_reduction_params
from torch import nn, Tensor


# def computeWeight(preds, psueduo, labm):
#     Kl_loss = KL_div()
#     loss_list = []
#     wk = []
#     for i in range(len(preds)):
#         loss_s = torch.zeros((preds[i].shape[0]))
#         loss = (Kl_loss(preds[i], psueduo)).data
#         for i in range(len(loss)):
#             loss_s[i] = loss[i].sum()
#         loss_list.append(loss_s)
#     for loss_one in loss_list:
#         ori_wk = (1 - 1 / labm * loss_one).to('cpu')
#         wk_one = torch.where(ori_wk.data < 0, torch.tensor([1e-16], dtype=torch.float).to('cpu'), ori_wk)
#         wk.append(wk_one)
#     return wk

# class KL_div(nn.Module):
#
#     def __init__(self, eps=1e-16, weight: Union[List[float], Tensor] = None, verbose=True):
#         super().__init__()
#         self._eps = eps
#         self._weight: Optional[Tensor] = weight
#         if weight is not None:
#             assert isinstance(weight, (list, Tensor)), type(weight)
#             if isinstance(weight, list):
#                 assert assert_list(lambda x: isinstance(x, (int, float)), weight)
#                 self._weight = torch.Tensor(weight).float()
#             else:
#                 self._weight = weight.float()
#             # normalize weight:
#             self._weight = self._weight / self._weight.sum()
#
#     def forward(self, prob: Tensor, target: Tensor, **kwargs) -> Tensor:
#         if not kwargs.get("disable_assert"):
#             assert prob.shape == target.shape
#             assert simplex(prob), prob
#             assert not target.requires_grad
#             assert prob.requires_grad
#         b, c, *hwd = target.shape
#         kl = (-target * torch.log((prob + self._eps) / (target + self._eps)))
#         if self._weight is not None:
#             assert len(self._weight) == c
#             weight = self._weight.expand(b, *hwd, -1).transpose(-1, 1).detach()
#             kl *= weight.to(kl.device)
#         kl = kl.sum(0)
#         return kl


def computeWeight(preds, psueduo, labm):
    Kl_loss = KL_div()
    wk = []
    for i in range(len(preds)):
        loss_wk = (Kl_loss(preds[i], psueduo)).data
        wk.append(loss_wk)
    wklist = []
    for i in range(len(wk)):
        wk_one = torch.where(wk[i] == 0, torch.tensor([1e-3], dtype=torch.float).to('cpu'), wk[i])
        wklist.append(wk_one)
    wwk = []
    for i in range(len(wklist)):
        ori_wk = (1 - 1 / labm * wklist[i]).to('cpu')
        wwk.append(ori_wk)
    wk_total = []
    for i in range(len(wwk)):
        wk_1 = torch.where(wwk[i] < 0, torch.tensor([1e-5], dtype=torch.float).to('cpu'), wwk[i])
        wk_total.append(wk_1)
    return wk_total


class KL_div(nn.Module):
    def __init__(self, eps=1e-16, weight: Union[List[float], Tensor] = None, verbose=True):
        super().__init__()
        self._eps = eps
        self._weight: Optional[Tensor] = weight
        if weight is not None:
            assert isinstance(weight, (list, Tensor)), type(weight)
            if isinstance(weight, list):
                assert assert_list(lambda x: isinstance(x, (int, float)), weight)
                self._weight = torch.Tensor(weight).float()
            else:
                self._weight = weight.float()
            # normalize weight:
            self._weight = self._weight / self._weight.sum()

    def forward(self, prob: Tensor, target: Tensor, **kwargs) -> Tensor:
        if not kwargs.get("disable_assert"):
            assert prob.shape == target.shape
            assert simplex(prob), prob
            assert not target.requires_grad
            assert prob.requires_grad
        b, c, *hwd = target.shape
        kl = (-target * torch.log((prob + self._eps) / (target + self._eps))).sum(1)
        return kl