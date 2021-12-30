import numpy as np
from utils.LRScheduler.ExpDecayLR import ExpDecay
from utils.LRScheduler.CosineAnnelingLR import CosineAnneling

# To get the LR plot against number of epochs as per the training config
# python LRHelper.py --op_path "./summaries/"

class LRHelper:
    def __init__(self, config):
        # config : Dict contains configuration regarding learning rate
        self.lr_scheduler = config.lr_scheduler

        if self.lr_scheduler == 'exp_decay':
            self.lr_class = ExpDecay(config)
        elif self.lr_scheduler == 'cosine_annealing':
            self.lr_class = CosineAnneling(config)
        else:
            raise NotImplementedError('Invalid lr_scheduler called.')

    
    def step(self, g_step, opt):
        lr = self.lr_class.step(g_step, opt)
        return lr


    def lr(self, g_step):
        return self.lr_class.get_lr(g_step)


    def plot_lr(self, op_path, eps):
        self.lr_class.plot_lr(op_path, eps)                       



if __name__ == '__main__':
    import argparse
    import config_training

    parser = argparse.ArgumentParser(description='Plot LR Curve.')
    parser.add_argument('--op_path', action='store', type=str, help='Output folder where LR tensorboard summary to be stored.')

    args = parser.parse_args()

    lr_handler = LRHelper(config_training)
    lr_handler.plot_lr(args.op_path, config_training.epochs)
