import numpy as np
from utils.LRScheduler.LRScheduler import LearningRateScheduler


class CosineAnneling(LearningRateScheduler):
    def __init__(self, config):
        super().__init__(config)
        self.cosine_iters = self.config.steps_per_epoch * self.config.epochs - self.config.burn_in_steps
            

    def get_lr(self, g_step):
        # we will scale lr from 0 to 1e-3 in first 3 epochs and then exp decay for rest of the training.
        if g_step < self.config.burn_in_steps:
            lr = (self.config.init_lr) * (g_step / self.config.burn_in_steps)     # Linear Scaling
            return lr
        else:
            # For exponential decay learning rate uncomment below line and comment subsequent lines.
            return 0 + (self.config.init_lr - 0) * 0.5 * (1 + np.cos(np.pi * (g_step - self.config.burn_in_steps) / self.cosine_iters))
        
