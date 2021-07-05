from typing import List

import torch
from deepclustering.loss.kl_losses import _check_reduction_params
from deepclustering.utils import assert_list, simplex, reduce
from torch import nn, Tensor


class SimplexCrossEntropyLoss(nn.Module):
    def __init__(self, reduction="mean", eps=1e-16) -> None:
        super().__init__()
        _check_reduction_params(reduction)
        self._reduction = reduction
        self._eps = eps

    def forward(self, prob: Tensor, target: Tensor, **kwargs) -> Tensor:
        if not kwargs.get("disable_assert"):
            assert not target.requires_grad
            assert prob.requires_grad
            assert prob.shape == target.shape
            assert simplex(prob)
            assert simplex(target)
        b, c, *_ = target.shape
        ce_loss = -target * torch.log(prob + self._eps)
        if self._reduction == "mean":
            return ce_loss.mean()
        elif self._reduction == "sum":
            return ce_loss.sum()
        else:
            return ce_loss


class Entropy(nn.Module):
    r"""General Entropy interface

    the definition of Entropy is - \sum p(xi) log (p(xi))

    reduction (string, optional): Specifies the reduction to apply to the output:
    ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
    ``'mean'``: the sum of the output will be divided by the number of
    elements in the output, ``'sum'``: the output will be summed.
    """

    def __init__(self, reduction="mean", eps=1e-16):
        super().__init__()
        _check_reduction_params(reduction)
        self._eps = eps
        self._reduction = reduction

    def forward(self, input: Tensor) -> Tensor:
        assert input.shape.__len__() >= 2
        b, _, *s = input.shape
        # assert simplex(input), f"Entropy input should be a simplex"
        e = -(input * (input + self._eps).log())
        e = e.sum(1)
        assert e.shape == torch.Size([b, *s])
        if self._reduction == "mean":
            return e.mean()
        elif self._reduction == "sum":
            return e.sum()
        else:
            return e


class JSD_div(nn.Module):
    """
    general JS divergence interface
    :<math>{\rm JSD}_{\pi_1, \ldots, \pi_n}(P_1, P_2, \ldots, P_n) = H\left(\sum_{i=1}^n \pi_i P_i\right) - \sum_{i=1}^n \pi_i H(P_i)</math>


    reduction (string, optional): Specifies the reduction to apply to the output:
        ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
        ``'mean'``: the sum of the output will be divided by the number of
        elements in the output, ``'sum'``: the output will be summed.
    """

    def __init__(self, reduction="mean", eps=1e-16):
        super().__init__()
        _check_reduction_params(reduction)
        self._reduction = reduction
        self._eps = eps
        self._entropy_criterion = Entropy(reduction=reduction, eps=eps)

    def forward(self, input: List) -> Tensor:
        num_models = len(input)
        if num_models < 4:
            for i in range(num_models):
                assert assert_list(
                    lambda x: simplex(x), input
                ), f"input tensor should be a list of simplex."
                assert assert_list(
                    lambda x: x.shape == input[i][0].shape, input[i]
                ), "input tensor should have the same dimension"

        mean_prob = reduce(lambda x, y: x + y, input) / len(input)
        f_term = self._entropy_criterion(mean_prob)
        mean_entropy = sum(list(map(lambda x: self._entropy_criterion(x), input))) / len(input)
        assert f_term.shape == mean_entropy.shape

        return f_term, mean_entropy


class GJSD_div(nn.Module):
    """
    general JS divergence interface
    :<math>{\rm JSD}_{\pi_1, \ldots, \pi_n}(P_1, P_2, \ldots, P_n) = H\left(\sum_{i=1}^n \pi_i P_i\right) - \sum_{i=1}^n \pi_i H(P_i)</math>


    reduction (string, optional): Specifies the reduction to apply to the output:
        ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
        ``'mean'``: the sum of the output will be divided by the number of
        elements in the output, ``'sum'``: the output will be summed.
    """

    def __init__(self, reduction="mean", eps=1e-16):
        super().__init__()
        _check_reduction_params(reduction)
        self._reduction = reduction
        self._eps = eps
        self._entropy_criterion = Entropy(reduction='none', eps=eps)

    # def forward(self, input: List, wk) -> Tensor:
    #     term1 = []
    #     term2 = []
    #     for i in range(len(input)):
    #         one_ent = 0
    #         weight_entropylist = []
    #         ent = self._entropy_criterion(input[i])
    #         for k in range(len(input[i])):
    #             one_ent = one_ent + wk[i][k] * ent[k].mean()
    #             weight_entropy = (wk[i][k] * input[i][k]).unsqueeze(0)
    #             weight_entropylist.append(weight_entropy)
    #         term_one = torch.cat([weight_ent for weight_ent in weight_entropylist], dim=0)
    #         term1.append(term_one)
    #         H_one = one_ent/len(input[i])
    #         term2.append(H_one)
    #     mean_prob = reduce(lambda x, y: x + y, term1) / len(input)
    #     f_term = self._entropy_criterion(mean_prob).mean()
    #     mean_entropy = sum(term2) / len(input)
    #
    #     return f_term, mean_entropy

    def forward(self, input: List, wk) -> Tensor:
        ent_list = []
        weight_preds_sum = 0
        for i in range(len(input)):
            ent = self._entropy_criterion(input[i])
            ent_list.append(ent)

            weight_preds = (wk[i]/(wk[0]+wk[1])).unsqueeze(1) * input[i]
            weight_preds_sum = weight_preds_sum + weight_preds

        term1 = self._entropy_criterion(weight_preds_sum)

        term2 = 0
        for i in range(len(ent_list)):
            term2 = term2 + wk[i] * ent_list[i]

        return term1.mean(), (term2/2).mean()