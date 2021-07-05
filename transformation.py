import torch
from torch import Tensor
from deepclustering.utils import assert_list
import numpy as np


class TensorRandomFlip:
    def __init__(self, axis=None) -> None:
        if isinstance(axis, int):
            self._axis = [axis]
        elif isinstance(axis, (list, tuple)):
            assert_list(lambda x: isinstance(x, int), axis), axis
            self._axis = axis
        else:
            raise ValueError(str(axis))

    def __call__(self, tensor: Tensor):
        new_tensor = torch.zeros_like(tensor)
        for i in range(len(tensor)):
            new_tensor[i] = tensor[i].flip(self._axis)
        return new_tensor

    def __repr__(self):
        return f"{self.__class__.__name__} \n axis={self._axis}."


def transforms_for_rot(stu_inputs):
    rot_mask = np.random.randint(0, 4, stu_inputs.shape[0])
    # rot_mask = [0,1,2,3,0,1,2,3]
    for idx in range(stu_inputs.shape[0]):
        stu_inputs[idx] = torch.rot90(stu_inputs[idx], int(rot_mask[idx]), dims=[1,2])
    return stu_inputs, rot_mask


def transforms_back_rot(stu_output, rot_mask):
    stu_outputs = torch.zeros_like(stu_output)
    for idx in range(stu_output.shape[0]):
        stu_outputs[idx] = torch.rot90(stu_output[idx], int(rot_mask[idx]), dims=[2,1])
    return stu_outputs
