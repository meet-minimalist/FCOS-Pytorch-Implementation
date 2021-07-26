
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class LearningRateScheduler:
    def __init__(self):
        pass

    def get_lr(self, g_step):
        # To enable this function as a virtual function exception can be raised if it is not implemented in child class.
        raise NotImplementedError('Child class should implement get_lr function.')


    def step(self, g_step, opt):
        lr = self.get_lr(g_step)

        for grp in opt.param_groups:
            grp['lr'] = lr

        return lr
        
    
    def plot_lr(self, op_path, eps, steps_per_eps):
        lr_sum_writer = SummaryWriter(op_path)

        for e in tqdm(range(eps)):
            for s in range(steps_per_eps):
                if (s+1) % 10 == 0:
                    g_step = steps_per_eps * e + s
                    lr = self.get_lr(g_step)
                    lr_sum_writer.add_scalar('lr', lr, g_step)