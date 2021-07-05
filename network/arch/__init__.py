from deepclustering.arch import _register_arch

from experimental_check.network.segnet import SegNet
from experimental_check.network.unet import UNet

_register_arch("segnet", SegNet)
_register_arch("UNnet", UNet)