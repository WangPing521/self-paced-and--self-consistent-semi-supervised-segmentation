import os
from typing import Iterable, Union

from torch import Tensor
import warnings
from pathlib import Path
from skimage.io import imsave
import numpy as np


def save_images(segs: Tensor, names: Iterable[str], root: Union[str, Path], mode: str, iter: int) -> None:
    (b, w, h) = segs.shape  # type: Tuple[int, int,int] # Since we have the class numbers, we do not need a C axis
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        for seg, name in zip(segs, names):

            save_path = Path(root, f"iter{iter:03d}", mode, name[7:10], name).with_suffix(".png")

            save_path.parent.mkdir(parents=True, exist_ok=True)

            imsave(str(save_path), seg.cpu().numpy().astype(np.uint8))