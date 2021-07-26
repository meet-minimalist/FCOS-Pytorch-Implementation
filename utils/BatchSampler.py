
# Ref. : This implementation of multi scale training has been taken from
#      : https://github.com/CaoWGG/multi-scale-training/blob/4f019b49d30a127cf763796ec9e1bd8bf5ab8747/batch_sampler.py#L4
#      : http://www.ericscuccimarra.com/blog/pytorch-multi-scale

import numpy as np
from torch.utils.data import Sampler

class BatchSampler(object):
    def __init__(self, sampler, batch_size, drop_last, multiscale_step=None, img_sizes=None):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        if multiscale_step is not None and multiscale_step < 1 :
            raise ValueError("multiscale_step should be > 0, but got "
                             "multiscale_step={}".format(multiscale_step))
        if multiscale_step is not None and img_sizes is None:
            raise ValueError("img_sizes must a list, but got img_sizes={} ".format(img_sizes))

        self.multiscale_step = multiscale_step
        self.img_sizes = img_sizes

    def __iter__(self):
        num_batch = 0
        batch = []
        size = self.img_sizes[0]
        for idx in self.sampler:
            batch.append([idx, size])
            if len(batch) == self.batch_size:
                yield batch
                num_batch+=1
                batch = []
                if self.multiscale_step and num_batch % self.multiscale_step == 0 :
                    size_idx = np.random.choice(len(self.img_sizes))
                    size = self.img_sizes[size_idx]
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size