import config
import numpy as np
from utils.LRScheduler.LRScheduler import LearningRateScheduler


class CosineAnneling(LearningRateScheduler):
    def __init__(self):
        super().__init__()
        self.cosine_iters = config.steps_per_epoch * config.epochs - config.burn_in_steps
            

    def get_lr(self, g_step):
        # we will scale lr from 0 to 1e-3 in first 3 epochs and then exp decay for rest of the training.
        if g_step < config.burn_in_steps:
            lr = (config.init_lr) * (g_step / config.burn_in_steps)     # Linear Scaling
            return lr
        else:
            # For exponential decay learning rate uncomment below line and comment subsequent lines.
            return 0 + (config.init_lr - 0) * 0.5 * (1 + np.cos(np.pi * (g_step - config.burn_in_steps) / self.cosine_iters))
        
