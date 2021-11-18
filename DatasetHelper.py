
import cv2
import csv
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import RandomSampler
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

import config
from utils.transforms.to_tensor import ToTensorOwn
from utils.transforms.normalize import Normalize
from utils.transforms.center_crop import CenterCrop
from utils.transforms.augmenter import Augmenter

from utils.CollateFunc import MergeVariableLenBBox
from utils.BatchSampler import BatchSampler


class VOCDataset(Dataset):
    def __init__(self, csv_path, num_classes, augment=False, basic_transforms=None, augment_transforms=None):
        
        with open(csv_path, 'r') as file:
            csv_file = csv.reader(file)
            self.ann_data = list(csv_file)
            # contains list of list. Each sub-list contains img_path and bbox details.            

        np.random.shuffle(self.ann_data)

        self.num_classes = num_classes
        self.basic_transforms = basic_transforms
        self.augment_transforms = augment_transforms
        self.augment = augment


    def __len__(self):
        return len(self.ann_data)


    def __getitem__(self, idx_data):
        if isinstance(idx_data, (tuple, list)):
            # BatchSampler function will call this function with [tuple, list]
            idx, input_size = idx_data
        else:
            # set the default image size here
            input_size = config.input_size
            idx = idx_data
        
        ann_line = self.ann_data[idx]
        # [img_path, img_h, img_w, \
            # xmin_1, ymin_1, xmax_1, ymax_1, cls_id_1, 
            # xmin_2, ymin_2, ...]

        img_path, _, _ = ann_line[:3]
        
        num_bboxes = int(len(ann_line[3:]) // 5)
        bboxes = []
        for i in range(num_bboxes):
            x1, y1, x2, y2, cls_id = [int(c) for c in ann_line[5*i+3 : 5*(i+1)+3]]
            bb = BoundingBox(x1, y1, x2, y2, label=cls_id)
            bboxes.append(bb)
        bbs = BoundingBoxesOnImage(bboxes, shape=cv2.imread(img_path).shape)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        sample = {'image' : img, 'bbox' : bbs}

        if self.augment:
            sample = self.augment_transforms(sample)
        
        # Mandatory transformations like resizing image to same size, normalizing the image and converting to tensor.
        sample = self.basic_transforms([sample, input_size])

        return sample



basic_transforms = transforms.Compose([
                        CenterCrop(),
                        ToTensorOwn(),             # Custom ToTensor transform, converts to CHW from HWC only
                        Normalize(config.normalization_type),
                    ])

augment_transforms = Augmenter()


def get_train_loader():
    train_set = VOCDataset(config.train_csv_path, num_classes=config.num_classes, \
                                augment=True, basic_transforms=basic_transforms, augment_transforms=augment_transforms)

    train_loader = torch.utils.data.DataLoader(
                            dataset=train_set, \
                            batch_sampler=BatchSampler(RandomSampler(train_set), \
                                                       batch_size=config.batch_size, \
                                                       # shuffle=True, \ # Not required as we used RandomSampler
                                                       drop_last=False, \
                                                       multiscale_step=config.multiscale_step, \
                                                       img_sizes=config.multiscale_input_sizes), \
                            collate_fn=MergeVariableLenBBox(),
                            num_workers=1, pin_memory=True, \
                            prefetch_factor=2, \
                            # persistent_workers=True)
                            persistent_workers=False)
    # For BatchSampler and MergeVariableLenBBox references check the respective .py file.
    # persistent_workers and pin_memory both cant be set to true at the same time due to some bug.
    # Ref: https://github.com/pytorch/pytorch/issues/48370

    # For windows num_workers should be set to 0 due to some know issue. In ubuntu it works fine.
    # Ref: https://github.com/pytorch/pytorch/issues/4418#issuecomment-354614162
    return train_loader


def get_test_loader():
    test_set = VOCDataset(config.test_csv_path, num_classes=config.num_classes, \
                                augment=False, basic_transforms=basic_transforms)

    test_loader = torch.utils.data.DataLoader(
                            dataset=test_set, \
                            batch_size=config.batch_size, \
                            shuffle=False, num_workers=1, pin_memory=True, \
                            drop_last=False, prefetch_factor=2, \
                            collate_fn=MergeVariableLenBBox(),
                            # persistent_workers=True)
                            persistent_workers=False)
    # persistent_workers and pin_memory both cant be set to true at the same time due to some bug.
    # Ref: https://github.com/pytorch/pytorch/issues/48370

    # For windows num_workers should be set to 0 due to some know issue. In ubuntu it works fine.
    # Ref: https://github.com/pytorch/pytorch/issues/4418#issuecomment-354614162
    return test_loader


"""
# Below code is for debugging purpose only.
if __name__ == "__main__":
        
    iterator = iter(get_train_loader())
    # iterator = iter(get_test_loader())
    
    for i in range(3):
        batch = next(iterator)

        train_img = batch['image']
        train_bboxes = batch['bbox']
        train_num_bb = batch['num_bbox']
        # print(type(train_img))
        # print(type(train_bboxes))
        # print(train_img.shape)
        # print(train_bboxes.shape)
        # print(train_num_bb)
        
        for img, bboxes, num_bb in zip(train_img, train_bboxes, train_num_bb):
            img = img.numpy()
            bboxes = bboxes.numpy()
            num_bb = num_bb.numpy()
            
            img = np.transpose(img, (1, 2, 0))
            img =  ((img * 0.5) + 0.5) * 255
            img = np.uint8(np.clip(img, 0, 255))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            for bb in bboxes[:num_bb]:
                x1, y1, x2, y2, cls = [int(c) for c in bb]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

            import imutils
            img = imutils.resize(img, width=720)
            cv2.imshow('img', img)
            cv2.waitKey()
            cv2.destroyAllWindows()
        
        #exit()   
"""