#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result


class GeneralizedDiceLoss(nn.Module):

    def __init__(self, smooth=1, p=2):
        super(GeneralizedDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0]
        assert predict.shape[1] == target.shape[1], "predict & target batch size don't match"
        total_loss = 0
        for i in range(target.shape[0]):
            pred = predict[i].contiguous().view(predict[i].shape[0], -1)
            tar = target[i].contiguous().view(target[i].shape[0], -1)
            weights = (1.0 / (tar.sum(dim=1) * tar.sum(dim=1) + 1e-15))
            intersection = torch.sum(torch.mul(pred, tar), dim=1)
            dice_list = 2 * intersection * weights / ((torch.sum(pred.pow(self.p) + tar.pow(self.p), dim=1) * weights) + np.float64(self.smooth))
            loss = 1 - dice_list.mean()
            total_loss = total_loss + loss
        return total_loss


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self._dice = GeneralizedDiceLoss(**self.kwargs)

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        total_loss = self._dice(predict, target)
        return total_loss/target.shape[0]