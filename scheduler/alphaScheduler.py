from deepclustering.schedulers.customized_scheduler import WeightScheduler
import numpy as np


class Alpha_RampScheduler(WeightScheduler):
    def __init__(
        self, begin_epoch=0, max_epoch=10, min_value=0.0, max_value=1.0, ramp_mult=-5.0
    ):
        super().__init__()
        self.begin_epoch = int(begin_epoch)
        self.max_epoch = int(max_epoch)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.mult = float(ramp_mult)
        self.epoch = 0

    def step(self):
        self.epoch += 1

    @property
    def value(self):
        return self.get_lr(
            self.epoch,
            self.begin_epoch,
            self.max_epoch,
            self.min_value,
            self.max_value,
            self.mult,
        )

    @staticmethod
    def get_lr(epoch, begin_epoch, max_epochs, min_val, max_val, mult):
        if epoch < begin_epoch:
            return min_val
        elif epoch >= max_epochs:
            return max_val
        return max_val * np.exp(mult * (1.0 - float(epoch - begin_epoch) / (max_epochs - begin_epoch)) ** 2)