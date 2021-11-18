import config
import numpy as np
from utils.LRScheduler.ExpDecayLR import ExpDecay
from utils.LRScheduler.CosineAnnelingLR import CosineAnneling

# To get the LR plot against number of epochs as per the training config
# python LRHelper.py --op_path "./summaries/"

class LRHelper:
    def __init__(self):
        self.lr_scheduler = config.lr_scheduler
        if self.lr_scheduler == 'exp_decay':
            self.lr_class = ExpDecay()
        elif self.lr_scheduler == 'cosine_annealing':
            self.lr_class = CosineAnneling()
        else:
            raise NotImplementedError('Invalid lr_scheduler called.')

    
    def step(self, g_step, opt):
        lr = self.lr_class.step(g_step, opt)
        return lr


    def lr(self, g_step):
        return self.lr_class.get_lr(g_step)


    def plot_lr(self, op_path, eps, steps_per_eps):
        self.lr_class.plot_lr(op_path, eps, steps_per_eps)                       



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot LR Curve.')
    parser.add_argument('--op_path', action='store', type=str, help='Output folder where LR tensorboard summary to be stored.')

    args = parser.parse_args()

    lr_handler = LRHelper()
    lr_handler.plot_lr(args.op_path, config.epochs, config.steps_per_epoch)
