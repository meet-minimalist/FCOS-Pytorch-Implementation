
import numpy as np
import torch
import torchvision
import torch.nn as nn

cuda = torch.device('cuda:0')
cpu = torch.device("cpu:0")


class PostProcessor(nn.Module):
    def __init__(self, use_cuda=False, add_centerness_in_cls_prob=True, \
                    max_detection_boxes_num=1000, cls_score_threshold=0.05, \
                    nms_iou_threshold=0.60, num_classes=20):
        super(PostProcessor, self).__init__()
        self.use_cuda = use_cuda
        self.add_centerness_in_cls_prob = add_centerness_in_cls_prob
        self.max_detection_boxes_num = max_detection_boxes_num
        self.cls_score_threshold = cls_score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.num_classes = num_classes
        
        
    
    def forward(self, detection_op, strides):
        cls_probs, cnt_logits, reg_values = detection_op
        # cls_probs, cnt_logit, reg_values each will have a list of features having shape as below.
        # cls_probs : [[B x 81 x H x W], [B x 81 x H x W], ....]
        # cnt_logits: [[B x 1 x H x W], [B x 1 x H x W], ....]
        # reg_values: [[B x 4 x H x W], [B x 4 x H x W], ....]
        # strides : [8, 16, 32, 64, 128]

        comb_cls_probs = []
        comb_cnt_logits = []
        comb_reg_values = []
        comb_coordinates = []
        for cls_p, cnt_l, reg_v, stride in zip(cls_probs, cnt_logits, reg_values, strides):
            num_classes, feat_h, feat_w = cls_p.shape[1:]
            
            comb_cls_probs.append(torch.reshape(cls_p, [-1, num_classes, feat_h * feat_w]))
            comb_cnt_logits.append(torch.reshape(cnt_l, [-1, 1, feat_h * feat_w]))
            comb_reg_values.append(torch.reshape(reg_v, [-1, 4, feat_h * feat_w]))


            grid_y = torch.arange(0, feat_h * stride, stride, dtype=torch.float32)
            grid_x = torch.arange(0, feat_w * stride, stride, dtype=torch.float32)
            
            grid_y, grid_x = torch.meshgrid(grid_y, grid_x)
            coords = torch.stack([torch.reshape(grid_x, [-1]), \
                                    torch.reshape(grid_y, [-1])], -1) + stride // 2
            coords = torch.unsqueeze(coords, 0)
            if self.use_cuda:
                coords = coords.to(cuda, non_blocking=True)
            # coords : [1 x H*W x 2] : center-points of each grid-cell : [x, y] at last dim.
            #                        : This coords are now in original image space i.e. 800x1024

            comb_coordinates.append(coords)

        comb_cls_probs = torch.cat(comb_cls_probs, dim=2)       # [B x 81 x sum(H*W)]
        comb_cnt_logits = torch.cat(comb_cnt_logits, dim=2)     # [B x 1 x sum(H*W)]
        comb_reg_values = torch.cat(comb_reg_values, dim=2)     # [B x 4 x sum(H*W)]


        comb_coordinates = torch.cat(comb_coordinates, dim=1)   # [1 x sum(H*W) x 2]
        comb_coordinates = torch.Tensor.permute(comb_coordinates, (0, 2, 1))   # [1 x 2 x sum(H*W)]

        comb_cnt_probs = torch.sigmoid(comb_cnt_logits)         # [B x 1 x sum(H*W)]
        
        comb_cls_scores, comb_cls_classes = torch.max(comb_cls_probs[:, 1:, :], dim=1)
        # Ignoring 0th class as it is background class
        # comb_cls_scores   : [B x sum(H*W)]    : probability
        # comb_cls_classes  : [B x sum(H*W)]    : class-id
        
        if self.add_centerness_in_cls_prob:
            comb_cls_scores = torch.sqrt(comb_cls_scores * comb_cnt_probs[:, 0, :])
            # [B x sum(H*W)]

        x1y1 = comb_coordinates - comb_reg_values[:, 0:2, :]    # [B x 2 x sum(H*W)]
        x2y2 = comb_coordinates + comb_reg_values[:, 2:4, :]    # [B x 2 x sum(H*W)]
        bboxes = torch.cat([x1y1, x2y2], dim=1)                 # [B x 4 x sum(H*W)]

        max_bboxes = min(self.max_detection_boxes_num, comb_cls_scores.shape[-1])
        topk_values, topk_idx = torch.topk(comb_cls_scores, max_bboxes, dim=-1)      # [B x max_bboxes]

        batch_predictions = []

        for b in range(comb_cls_scores.shape[0]):   # Iterate for each element in batch 
            _bb_scores = comb_cls_scores[b][topk_idx[b]]     # [max_bboxes]
            mask = _bb_scores >= self.cls_score_threshold
            
            _bb_scores = _bb_scores[mask]                           # [N]
            _bb_classes = comb_cls_classes[b][topk_idx[b]][ mask]   # [N]
            _bb_coords = bboxes[b][:, topk_idx[b]][:, mask]         # [4 x N]

            _bb_coords = _bb_coords.permute(1, 0)                   # [N x 4]

            idx = torchvision.ops.batched_nms(_bb_coords,               # [N x 4]
                                              _bb_scores,               # [N]
                                              _bb_classes,              # [N]
                                              self.nms_iou_threshold)

            _bb_scores_filtered = _bb_scores[idx]        # [N]
            _bb_classes_filtered = _bb_classes[idx]      # [N]
            _bb_coords_filtered = _bb_coords[idx]        # [N x 4]


            predictions = torch.cat([_bb_coords_filtered, 
                                     _bb_scores_filtered.view(-1, 1), 
                                     _bb_classes_filtered.view(-1, 1).to(torch.float32)], dim=1) # [N x 6]
            # predictions : [N x 6] : [x1, y1, x2, y2, cls_prob, cls_id]
    
            batch_predictions.append(predictions)


        return batch_predictions
