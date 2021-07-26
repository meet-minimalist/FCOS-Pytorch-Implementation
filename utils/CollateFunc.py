
# Ref. : This implementation to accomodate variable length bboxes so that they can be concatted in a single batch.
#      : https://www.codefull.net/2018/11/use-pytorchs-dataloader-with-variable-length-sequences-for-lstm-gru/

import torch

class MergeVariableLenBBox:
    def __call__(self, batch):
        max_bbox_in_batch = 0
        for sample_dict in batch:
            bbox = sample_dict['bbox']
            if max_bbox_in_batch < bbox.shape[0]:
                max_bbox_in_batch = bbox.shape[0]
            
        batch_img = []
        batch_bbox = []
        batch_num_bboxes = []
        for sample_dict in batch:
            img, bbox = sample_dict['image'], sample_dict['bbox']
            pad_len = max_bbox_in_batch - bbox.shape[0]
            
            num_bboxes = bbox.shape[0]
            bbox = torch.nn.functional.pad(bbox, pad=(0, 0, 0, pad_len))        
            
            batch_num_bboxes.append(num_bboxes)
            batch_img.append(img)
            batch_bbox.append(bbox)
            
        batch_img = torch.stack(batch_img, dim=0)
        batch_bbox = torch.stack(batch_bbox, dim=0)
        batch_num_bboxes = torch.tensor(batch_num_bboxes)

        return {'image' : batch_img, 'bbox' : batch_bbox, 'num_bbox' : batch_num_bboxes}