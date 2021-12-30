import os
import torch
from numpy.core.numeric import NaN

class CheckpointHandler:
    def __init__(self, ckpt_path, max_to_keep=3):
        self.ckpt_path = ckpt_path
        self.max_to_keep = max_to_keep
        self.ckpt_path_history = []

    def save(self, checkpoint_state):
        eps = checkpoint_state['epoch']
        try:
            test_loss = checkpoint_state['test_loss']
            cur_ckpt_path = os.path.splitext(self.ckpt_path)[0] + "_eps_{}_test_loss_{:.4f}".format(eps, test_loss) + os.path.splitext(self.ckpt_path)[1]
        except:
            train_loss = checkpoint_state['train_loss']
            cur_ckpt_path = os.path.splitext(self.ckpt_path)[0] + "_eps_{}_train_loss_{:.4f}".format(eps, train_loss) + os.path.splitext(self.ckpt_path)[1]

        torch.save(checkpoint_state, cur_ckpt_path)

        self.ckpt_path_history.append(cur_ckpt_path)

        if len(self.ckpt_path_history) > 3:
            # Remove old checkpoints as per max_to_keep arguments.
            remove_ckpt_path = self.ckpt_path_history.pop(0)
            os.remove(remove_ckpt_path)
