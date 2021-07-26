import torch
from torchvision import transforms


class Normalize(object):
    def __init__(self, model_type):
        assert model_type in ['imagenet', 'div_255']
        self.model_type = model_type
        if model_type == 'imagenet':
            self.imgnet_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    def __call__(self, sample):
        if self.model_type == 'div_255':
            sample['image'] = (sample['image'] / 255.0).type(torch.float32)
        elif self.model_type == 'imagenet':
            sample['image'] = (sample['image'] / 255.0).type(torch.float32)
            sample['image'] = self.imgnet_normalize(sample['image'].type(torch.float32))
        else:
            print("Normalization type not supported")

        return sample


