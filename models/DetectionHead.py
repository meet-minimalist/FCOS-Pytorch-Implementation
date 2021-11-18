import torch
import torch.nn as nn
from models.model_utils import CustomPadTensor


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor([init_value], dtype=torch.float32))
        
    def forward(self, x):
        return torch.exp(x * self.scale)
        

class DetectionBlock(nn.Module):
    def __init__(self, num_classes=81, fpn_features=256, use_group_norm=True, centerness_on_regression=True):
        super(DetectionBlock, self).__init__()

        self.pad_3x3 = CustomPadTensor(k_size=(3, 3), padding='SAME')
                
        self.centerness_on_regression = centerness_on_regression
        
        self.cls_branch = []
        self.reg_branch = []
        for _ in range(4):
            self.cls_branch.append(self.pad_3x3)
            self.cls_branch.append(nn.Conv2d(fpn_features, fpn_features, kernel_size=3, stride=1, bias=not use_group_norm))
            if use_group_norm:
                self.cls_branch.append(nn.GroupNorm(32, fpn_features))
            self.cls_branch.append(nn.ReLU())
        
            self.reg_branch.append(self.pad_3x3)
            self.reg_branch.append(nn.Conv2d(fpn_features, fpn_features, kernel_size=3, stride=1, bias=not use_group_norm))
            if use_group_norm:
                self.reg_branch.append(nn.GroupNorm(32, fpn_features))
            self.reg_branch.append(nn.ReLU())

        self.cls_branch = nn.Sequential(*self.cls_branch)
        self.reg_branch = nn.Sequential(*self.reg_branch)

        self.cls_conv = nn.Conv2d(fpn_features, num_classes, kernel_size=3, stride=1, bias=True)
        self.centerness_conv = nn.Conv2d(fpn_features, 1, kernel_size=3, stride=1, bias=True)
        self.reg_conv = nn.Conv2d(fpn_features, 4, kernel_size=3, stride=1, bias=True)

        # Ref. : https://github.com/tianzhi0549/FCOS/issues/33
        self.scale_mul = nn.ModuleList([Scale(1.0) for _ in range(5)])

                        
    def forward(self, feat_list):
        # p3_3x3 : [B x 256 x 100 x 128]
        # p4_3x3 : [B x 256 x 50 x 64]
        # p5_3x3 : [B x 256 x 25 x 32] 
        # p6_3x3 : [B x 256 x 13 x 16]
        # p7_3x3 : [B x 256 x 7 x 8]
        
        combined_cls_probs = []
        combined_centerness_logits = []
        combined_reg_values = []
        
        for i, feat in enumerate(feat_list):
            cls_feat = self.cls_branch(feat)
            reg_feat = self.reg_branch(feat)
        
            cls_logits = self.cls_conv(self.pad_3x3(cls_feat))      # [B x 81 x H x W]
            cls_probs = torch.sigmoid(cls_logits)
            
            if self.centerness_on_regression:
                centerness_logits = self.centerness_conv(self.pad_3x3(reg_feat))    # [B x 1 x H x W]
            else:
                centerness_logits = self.centerness_conv(self.pad_3x3(cls_feat))    # [B x 1 x H x W]
            
            reg_values = self.scale_mul[i](self.reg_conv(self.pad_3x3(reg_feat)))   # [B x 4 x H x W]

            combined_cls_probs.append(cls_probs)
            combined_centerness_logits.append(centerness_logits)
            combined_reg_values.append(reg_values)
            
        return combined_cls_probs, combined_centerness_logits, combined_reg_values