
import torch
import torch.nn as nn


class FCOSLoss(nn.Module):
    def __init__(self, limit_range, central_sampling=True, central_sampling_radius=1.5, \
                        strides=[8, 16, 32, 64, 128], regression_loss_type='giou', num_classes=81):
        super(FCOSLoss, self).__init__()
        self.limit_range = limit_range
        self.strides = strides
        self.num_classes = num_classes
        self.central_sampling = central_sampling
        self.regression_loss_type = regression_loss_type
        self.central_sampling_radius = central_sampling_radius
        
        
    def forward(self, prediction, batch_bboxes):
        # prediction   : list of 3 tensors cls_probs, cnt_logits, reg_values
        # batch_bboxes : [B x M x 5] --> [x1, y1, x2, y2, cls_id]
        
        cls_probs_pred, cnt_logits_pred, reg_values_pred = prediction
        # cls_probs_pred  : [[B x 81 x H x W], [B x 81 x H x W], ...]
        # cnt_logits_pred : [[B x 1 x H x W], [B x 1 x H x W], ...]
        # reg_values_pred : [[B x 4 x H x W], [B x 4 x H x W], ...]
        
        comb_cls_probs_pred = []
        comb_cnt_logits_pred = []
        comb_reg_values_pred = []
        
        comb_cls_probs_target = []
        comb_cnt_logits_target = []
        comb_reg_values_target = []

        for i, (cls_p, cnt_p, reg_p) in enumerate(zip(cls_probs_pred, cnt_logits_pred, reg_values_pred)):
            num_classes, feat_h, feat_w = cls_p.shape[1:4]
            cls_target, cnt_target, reg_target = \
                self.generate_targets(feat_h, feat_w, batch_bboxes, \
                                        self.strides[i], self.limit_range[i])    
            # cls_target : [B x H*W x 81]
            # cnt_target : [B x H*W x 1]
            # reg_target : [B x H*W x 4]
            
            # TODO : Cant we just calculate loss for each scale inside this loop rather than 
            #        Combining these tensors and computing together.
            cls_p = torch.reshape(cls_p, (-1, num_classes, feat_h*feat_w)).permute([0, 2, 1])      # [B x H*W x 81]
            cnt_p = torch.reshape(cnt_p, (-1, 1, feat_h*feat_w)).permute([0, 2, 1])                # [B x H*W x 1]
            reg_p = torch.reshape(reg_p, (-1, 4, feat_h*feat_w)).permute([0, 2, 1])                # [B x H*W x 4]
            
            comb_cls_probs_pred.append(cls_p)
            comb_cnt_logits_pred.append(cnt_p)
            comb_reg_values_pred.append(reg_p)
            
            comb_cls_probs_target.append(cls_target)
            comb_cnt_logits_target.append(cnt_target)
            comb_reg_values_target.append(reg_target)
            
        comb_cls_probs_target = torch.cat(comb_cls_probs_target, dim=1)      # [B x sum of all H*W x 81]
        comb_cnt_logits_target = torch.cat(comb_cnt_logits_target, dim=1)    # [B x sum of all H*W x 1]
        comb_reg_values_target = torch.cat(comb_reg_values_target, dim=1)    # [B x sum of all H*W x 4]
        
        comb_cls_probs_pred = torch.cat(comb_cls_probs_pred, dim=1)    # [B x sum of all H*W x 81]
        comb_cnt_logits_pred = torch.cat(comb_cnt_logits_pred, dim=1)  # [B x sum of all H*W x 1]
        comb_reg_values_pred = torch.cat(comb_reg_values_pred, dim=1)  # [B x sum of all H*W x 4]
        
        mask_pos = comb_cnt_logits_target > -1                         # [B x sum of all H*W x 1]
        cls_loss = self.compute_cls_loss(comb_cls_probs_pred, comb_cls_probs_target, mask_pos)
        cnt_loss = self.compute_centerness_loss(comb_cnt_logits_pred, comb_cnt_logits_target, mask_pos) 
        reg_loss = self.compute_coordinate_reg_loss(comb_reg_values_pred, comb_reg_values_target, mask_pos)
        return cls_loss, cnt_loss, reg_loss
        
        
    def generate_targets(self, feat_h, feat_w, batch_bbox, \
                                stride, limit_range):
        # feat_h     : Height of the feature map
        # feat_w     : Width of the feature map
        # batch_bbox : [B x M x 5]  --> [x1, y1, x2, y2, cls_id]
        #            : Not all images in B are having M bboxes but the are padded to make them M
        #            : So that they can be stacked in a tensor.
        # stride : value of stride for this feature-map
        # limit_range : [min_val, max_val]. To filter most relevant bboxes out of these feature-map
        
        target_device = batch_bbox.device
        grid_y = torch.arange(0, feat_h * stride, stride, dtype=torch.float32)
        grid_x = torch.arange(0, feat_w * stride, stride, dtype=torch.float32)
        
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x)
        coords = torch.stack([torch.reshape(grid_x, [-1]), \
                                torch.reshape(grid_y, [-1])], -1) + stride // 2
        coords = torch.unsqueeze(coords, 0).to(target_device)
        # coords : [1 x H*W x 2] : center-points of each grid-cell : [x, y] at last dim.
        #                        : This coords are now in original image space i.e. 800x1024
        coords_x, coords_y = coords[:, :, 0], coords[:, :, 1]   # [1 x H*W]
        
        batch_bb_x1, batch_bb_y1, batch_bb_x2, batch_bb_y2 = \
            batch_bbox[:, :, 0], batch_bbox[:, :, 1], batch_bbox[:, :, 2], batch_bbox[:, :, 3]
        # [B x M]
        
        # coords_x   : [1 x H*W] --> [1 x H*W x 1]
        # batch_bb_x1: [B x M] --> [B x 1 x M]
        l_off = torch.unsqueeze(coords_x, 2) - torch.unsqueeze(batch_bb_x1, 1)
        r_off = torch.unsqueeze(batch_bb_x2, 1) - torch.unsqueeze(coords_x, 2)
        t_off = torch.unsqueeze(coords_y, 2) - torch.unsqueeze(batch_bb_y1, 1)
        b_off = torch.unsqueeze(batch_bb_y2, 1) - torch.unsqueeze(coords_y, 2)
        ltrb_off = torch.stack([l_off, t_off, r_off, b_off], -1)    # [B x H*W x M x 4]
        # [left_x, top_y, right_x, bottom_y]
        
        areas = (l_off + r_off) * (t_off + b_off)            # [B x H*W x M]
        # For each location we have relationg with All M bboxes.
        # But we will select only one bbox out of these M bboxes for which  
        # the area is lowest. 
        
        off_min = torch.min(ltrb_off, axis=-1)[0]           # [B x H*W x M]
        off_max = torch.max(ltrb_off, axis=-1)[0]           # [B x H*W x M]
        
        mask_feat_map_limit = (off_max > limit_range[0]) & (off_max <= limit_range[1])
        # [B x H*W x M]
        
        mask_in_gtbbox = off_min > 0        # [B x H*W x M]
        # The bbox out of M which are padded, for such bbox this area will return
        # -ve and it will be the minimum area out of all the M bboxes. This would 
        # break the next logic. So to prevent it, this mask is introduced.
        
        if self.central_sampling:
            # Ref. : https://github.com/yqyao/FCOS_PLUS/issues/13#issuecomment-564823086
            img_level_radius = self.central_sampling_radius * stride
        
            gtbbox_xc = (batch_bb_x1 + batch_bb_x2) / 2     # [B x M]
            gtbbox_yc = (batch_bb_y1 + batch_bb_y2) / 2     # [B x M]
            
            # coords_x   : [1 x H*W] --> [1 x H*W x 1]
            # batch_bb_x1: [B x M] --> [B x 1 x M]
            center_x_off = torch.abs(torch.unsqueeze(coords_x, 2) - torch.unsqueeze(gtbbox_xc, 1))
            center_y_off = torch.abs(torch.unsqueeze(coords_y, 2) - torch.unsqueeze(gtbbox_yc, 1))
            center_xy_off = torch.stack([center_x_off, center_y_off], dim=-1)   # [B x H*W x M x 2]
            center_off_max = torch.max(center_xy_off, dim=-1)[0]                # [B x H*W x M] 
            mask_central_sampling = center_off_max < img_level_radius           # [B x H*W X M]

        else:
            mask_central_sampling = torch.ones_like(mask_in_gtbbox)     # [B x H*W X M]
        
        mask_positive_bbox = mask_feat_map_limit & mask_in_gtbbox & mask_central_sampling
        # [B x H*W x M]
        
        areas[~mask_positive_bbox] = 9999999            # [B x H*W x M]
        # We would make area for negative bboxes infinite so that the minimum area
        # will be computed from positive bboxes only.
        
        
        area_min_idx = torch.min(areas, dim=-1)[1]      # [B x H*W]
        
        # areas : [B x H*W x M]            
        ltrb_off_mask = torch.zeros_like(areas, dtype=torch.bool).scatter_(\
                                -1, area_min_idx.unsqueeze(dim=-1), 1)
        # This is the binary mask of shape [B x H*W x M] with value 1 for 
        # all the indexes mentioned by area_min_idx at the last dimension which is having
        # M values. So out of those M values at last dimension, only one value which is 
        # specified in area_min_idx will be made to 1.0 and rest will be kept as it is at 0.0
        
        reg_targets = ltrb_off[ltrb_off_mask]       # [B*H*W x 4]
        reg_targets = torch.reshape(reg_targets, (-1, feat_h * feat_w, 4))  # [B x H*W x 4]
        
        classes = torch.unsqueeze(batch_bbox[:, :, 4], dim=1)           # [B x 1 x M]
        classes = torch.broadcast_tensors(classes, areas.long())[0]     # [B x H*W x M]
        
        cls_targets = classes[ltrb_off_mask]                                # [B*H*W x 1]
        cls_targets = torch.reshape(cls_targets, (-1, feat_h * feat_w, 1))  # [B x H*W x 1]
        
        mask_positive_bbox_2 = mask_positive_bbox.long().sum(dim=-1)        # [B x H*W]
        mask_positive_bbox_2 = mask_positive_bbox_2 >= 1
    

        left_right_min = torch.min(reg_targets[:, :, 0], reg_targets[:, :, 2])      # [B x H*W]
        left_right_max = torch.max(reg_targets[:, :, 0], reg_targets[:, :, 2])      # [B x H*W]
        top_bottom_min = torch.min(reg_targets[:, :, 1], reg_targets[:, :, 3])      # [B x H*W]
        top_bottom_max = torch.max(reg_targets[:, :, 1], reg_targets[:, :, 3])      # [B x H*W]
        cnt_targets = torch.unsqueeze(torch.sqrt((left_right_min * top_bottom_min) / 
                            (left_right_max * top_bottom_max + 1e-9)), dim=-1)     # [B x H*W x 1]
        
        
        cls_targets[~mask_positive_bbox_2] = 0              # [B x H*W x 1]
        # Assigning label=0 which is background class, for all such locations 
        # which are negative i.e. locations which dont get associated to any bbox
        # print(cls_targets.shape)
        
        cls_targets = torch.nn.functional.one_hot(cls_targets[:, :, 0].long(), num_classes=self.num_classes)
        # [B x H*W x 1] ---> [B x H*W x 81]
        
        cnt_targets[~mask_positive_bbox_2] = -1             # [B x H*W x 1]
        reg_targets[~mask_positive_bbox_2] = -1             # [B x H*W x 4]
        return cls_targets, cnt_targets, reg_targets
    
    
    def compute_cls_loss(self, comb_cls_probs_pred, comb_cls_probs_target, mask_pos, 
                            alpha=0.25, gamma=2.0):
        # comb_cls_probs_pred   : [B x sum of all H*W x 81]
        # comb_cls_probs_target : [B x sum of all H*W x 81]
        # mask_pos              : [B x sum of all H*W x 1]
        
        assert comb_cls_probs_pred.shape[:2] == comb_cls_probs_target.shape[:2]
        
        num_pos = torch.sum(mask_pos, dim=[1, 2]).clamp_(min=1).float()     # [B]
        
        # Focal loss to be computed for all grid points
        # Positive as well as negative
        pt = comb_cls_probs_pred * comb_cls_probs_target + \
            (1 - comb_cls_probs_pred) * (1 - comb_cls_probs_target)        
        w = alpha * comb_cls_probs_target + (1 - alpha) * (1 - comb_cls_probs_target)
        focal_loss = (-1) * w * torch.pow((1 - pt), gamma) * torch.log(pt + 1e-10)
        focal_loss = focal_loss.sum(dim=[1, 2])         # [B]

        focal_loss = focal_loss / num_pos               # [B]
        return focal_loss.mean()                        # [1]
        
        
    def compute_centerness_loss(self, comb_cnt_logits_pred, comb_cnt_logits_target, mask_pos):
        # comb_cnt_logits_pred   : [B x sum of all H*W x 1]
        # comb_cnt_logits_target : [B x sum of all H*W x 1]
        # mask_pos               : [B x sum of all H*W x 1]
        
        num_pos = torch.sum(mask_pos, dim=[1, 2]).clamp_(min=1).float()     # [B]
        
        cnt_loss_batch = []
        batch_size = comb_cnt_logits_pred.shape[0]
        for i in range(batch_size):
            preds = comb_cnt_logits_pred[i][mask_pos[i]]          # [n]
            targets = comb_cnt_logits_target[i][mask_pos[i]]      # [n]
        
            cnt_loss = nn.functional.binary_cross_entropy_with_logits(input=preds, \
                                target=targets, reduction='none') # [n]
            cnt_loss = torch.sum(cnt_loss)                        # [1]
            cnt_loss_batch.append(cnt_loss)

        cnt_loss_batch = torch.stack(cnt_loss_batch, dim=0)       # [B]
        cnt_loss_batch = cnt_loss_batch / num_pos                 # [B]
        return cnt_loss_batch.mean()                              # [1]
        
        
    def compute_coordinate_reg_loss(self, comb_reg_values_pred, comb_reg_values_target, mask_pos):
        # comb_reg_values_pred   : [B x sum of all H*W x 4]
        # comb_reg_values_target : [B x sum of all H*W x 4]
        # mask_pos               : [B x sum of all H*W x 1]
        
        num_pos = torch.sum(mask_pos, dim=[1, 2]).clamp_(min=1).float()     # [B]
        
        reg_loss_batch = []
        batch_size = comb_reg_values_pred.shape[0]
        for i in range(batch_size):
            pred_pos = comb_reg_values_pred[i][mask_pos[i, :, 0]]           # [n, 4]
            target_pos = comb_reg_values_target[i][mask_pos[i, :, 0]]       # [n, 4]
            
            if self.regression_loss_type == 'iou':
                reg_loss = self.iou_loss(pred_pos, target_pos)  # [1]
            elif self.regression_loss_type == 'giou':
                reg_loss = self.giou_loss(pred_pos, target_pos) # [1]
            else:
                raise NotImplementedError(f"Regression Loss type: {self.regression_loss_type} is not supported.")    
            reg_loss_batch.append(reg_loss)
            
        reg_loss_batch = torch.stack(reg_loss_batch, dim=0)     # [B]
        reg_loss_batch = reg_loss_batch / num_pos
        return reg_loss_batch.mean()                            # [1]
    
    
    def iou_loss(self, pred_pos, target_pos):
        # pred_pos      : [n, 4]    : [left_x, top_y, right_x, bottom_y]
        # target_pos    : [n, 4]    : [left_x, top_y, right_x, bottom_y]
        
        x1y1_intersection = torch.min(pred_pos[:, :2], target_pos[:, :2])   # [n, 2]
        x2y2_intersection = torch.min(pred_pos[:, 2:], target_pos[:, 2:])   # [n, 2]
        wh = (x1y1_intersection + x2y2_intersection).clamp(min=0)           # [n, 2]
        overlap_area = wh[:, 0] * wh[:, 1]                                  # [n]
        area1 = (pred_pos[:, 2] + pred_pos[:, 0]) \
                    * (pred_pos[:, 3] + pred_pos[:, 1])             # [n]
        area2 = (target_pos[:, 2] + target_pos[:, 0]) \
                    * (target_pos[:, 3] + target_pos[:, 1])         # [n]
        
        iou = overlap_area / (area1 + area2 - overlap_area + 1e-9)  # [n]
        
        iou_loss = -torch.log(iou.clamp(1e-9))                      # [n]
        
        return iou_loss.sum()                                       # [1]
        
    
    def giou_loss(self, pred_pos, target_pos):
        # pred_pos      : [n, 4]    : [left_x, top_y, right_x, bottom_y]
        # target_pos    : [n, 4]    : [left_x, top_y, right_x, bottom_y]
        
        x1y1_outer = torch.max(pred_pos[:, :2], target_pos[:, :2])   # [n, 2]
        x2y2_outer = torch.max(pred_pos[:, 2:], target_pos[:, 2:])   # [n, 2]
        wh_outer = (x1y1_outer + x2y2_outer).clamp(min=0)            # [n, 2]
        outer_area = wh_outer[:, 0] * wh_outer[:, 1]                 # [n]
        
        x1y1_intersection = torch.min(pred_pos[:, :2], target_pos[:, :2])   # [n, 2]
        x2y2_intersection = torch.min(pred_pos[:, 2:], target_pos[:, 2:])   # [n, 2]
        wh = (x1y1_intersection + x2y2_intersection).clamp(min=0)           # [n, 2]
        overlap_area = wh[:, 0] * wh[:, 1]                                  # [n]
        area1 = (pred_pos[:, 2] + pred_pos[:, 0]) \
                    * (pred_pos[:, 3] + pred_pos[:, 1])             # [n]
        area2 = (target_pos[:, 2] + target_pos[:, 0]) \
                    * (target_pos[:, 3] + target_pos[:, 1])         # [n]
        
        union = area1 + area2 - overlap_area                        # [n]
        iou = overlap_area / (union + 1e-9)                         # [n]

        giou_loss = 1 - iou + \
                        ((outer_area - union) / (outer_area + 1e-9))    # [n] 
    
        return giou_loss.sum()                                      # [1]
    
    
if __name__ == '__main__':
    import sys
    import numpy as np
    sys.path.append("../")
    import config_temp as config
    loss_fn = FCOSLoss(limit_range=config.limit_range, central_sampling=config.central_sampling, \
                        central_sampling_radius=config.central_sampling_radius, strides=config.strides, \
                        regression_loss_type=config.regression_loss_type, num_classes=config.num_classes)
    
    cls_probs_pred = []
    cnt_logits_pred = []
    reg_values_pred = []
    batch_bboxes = []
    for s in config.strides:
        f_h, f_w = config.input_size[0] // s, config.input_size[1] // s
        
        cls_probs_pred.append(torch.randn(1, config.num_classes, f_h, f_w))
        cnt_logits_pred.append(torch.randn(1, 1, f_h, f_w))
        reg_values_pred.append(torch.randn(1, 4, f_h, f_w))
        
        rnd_num = np.random.randint(low=1, high=10)
        pad_len = 10 - rnd_num
        bbox = torch.randn(1, rnd_num, 5)
        bbox = torch.nn.functional.pad(bbox, pad=(0, 0, 0, pad_len))        
        batch_bboxes.append(bbox)
        
    prediction = [cls_probs_pred, cnt_logits_pred, reg_values_pred]
    
    batch_bboxes = torch.cat(batch_bboxes, dim=1)
    
    loss_fn(prediction, batch_bboxes)