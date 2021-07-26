import torch
import numpy as np

class ToTensorOwn(object):

    def __call__(self, sample):
        # This would receive only dict as CenterCrop is returning only dict instead of [dict, list]
        image, bbs = sample['image'], sample['bbox']

        bboxes = np.zeros(shape=[len(bbs.items), 5])
        for i, bb in enumerate(bbs.items):
            bboxes[i] = [bb.x1, bb.y1, bb.x2, bb.y2, bb.label]
            
        image = image.transpose(2, 0, 1)
        return {'image' : torch.from_numpy(image).type(torch.float32), 'bbox' : torch.tensor(bboxes, dtype=torch.float32)}