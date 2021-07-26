import config
import numpy as np
from utils.LRScheduler.LRScheduler import LearningRateScheduler


class ExpDecay(LearningRateScheduler):
    def __init__(self):
        super().__init__()
            

    def get_lr(self, g_step):
        # we will scale lr from 0 to 1e-3 in first 3 epochs and then exp decay for rest of the training.
        if g_step < config.burn_in_steps:
            lr = (config.init_lr) * (g_step / config.burn_in_steps)         # Linear Scaling
            #lr = config.init_lr * (g_step / config.burn_in_steps) ** 4      # Polynomial scaling
            return lr
        else:
            # For exponential decay learning rate uncomment below line and comment subsequent lines.
            return config.init_lr * np.exp( -(1 - config.lr_exp_decay) * (g_step - config.burn_in_steps) / config.steps_per_epoch)
        
