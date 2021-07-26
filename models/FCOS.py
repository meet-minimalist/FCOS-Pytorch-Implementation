
import numpy as np
import torch.nn as nn
from models.FPN import FPN
from models.DetectionHead import DetectionBlock


class FCOS(nn.Module):
    def __init__(self, backbone_model='resnet50', freeze_backend=[False, False, False, False], \
                        fpn_features=256, num_classes=81, use_det_head_group_norm=True, \
                        centerness_on_regression=True, use_gradient_checkpointing=False, \
                        weight_init_method='msra'):
        super(FCOS, self).__init__()
        self.backbone_model = backbone_model
        self.fpn_features = fpn_features
        self.freeze_backend = freeze_backend
        self.num_classes = num_classes
        self.use_det_head_group_norm = use_det_head_group_norm
        self.centerness_on_regression = centerness_on_regression
        self.weight_init_method = weight_init_method
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        if self.backbone_model == 'resnet50':
            from models.ResNet50 import ResNet50
            self.backbone = ResNet50(self.freeze_backend, self.use_gradient_checkpointing)
        elif self.backbone_model == 'resnet18':
            from models.ResNet18 import ResNet18
            self.backbone = ResNet18(self.freeze_backend, self.use_gradient_checkpointing)
        else:
            raise NotImplementedError('Invalid backbone selected.')
        
        if self.backbone_model == 'resnet50':
            self.fpn = FPN(pyramid_features=[512, 1024, 2048], features=self.fpn_features, \
                            use_gradient_checkpointing=self.use_gradient_checkpointing)
        elif self.backbone_model == 'resnet18':
            self.fpn = FPN(pyramid_features=[128, 256, 512], features=self.fpn_features, \
                            use_gradient_checkpointing=self.use_gradient_checkpointing)
        else:
            raise NotImplementedError('Invalid backbone selected.')
            
        self.detection = DetectionBlock(self.num_classes, self.fpn_features, \
                    self.use_det_head_group_norm, self.centerness_on_regression)
        self.__init_weights()
    
    
    def __init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if self.weight_init_method == 'xavier_normal':
                    nn.init.xavier_normal_(module.weight, gain=1.0)
                elif self.weight_init_method == 'msra':
                    nn.init.kaiming_normal_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
                else:
                    print("Unsupported weight init method.")
                    exit()
            
                if module.bias is not None:
                    # Ref : Classification branch last layer bias is initialized in same way as RetinaNet
                    #       See RetinaNet : Section 4.1 - Initialization
                    if 'cls_logits' in name:
                        pie = 0.01
                        nn.init.constant_(module.bias, -np.log((1-pie)/pie))
                    else:
                        nn.init.constant_(module.bias, 0)
                
            if isinstance(module, nn.GroupNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                
            if isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.running_var, 1)
                nn.init.constant_(module.running_mean, 0)
            
            
    def forward(self, x):    
        # x : [B x 3 x 800 x 1024]
        
        c3, c4, c5 = self.backbone(x)
        # c3 : [B x 512 x 100 x 128]
        # c4 : [B x 1024 x 50 x 64]
        # c5 : [B x 2048 x 25 x 32]
        
        p3, p4, p5, p6, p7 = self.fpn([c3, c4, c5])
        # p3 : [B x 256 x 100 x 128]
        # p4 : [B x 256 x 50 x 64]
        # p5 : [B x 256 x 25 x 32] 
        # p6 : [B x 256 x 12 x 16]
        # p7 : [B x 256 x 6 x 8]
        
        cls_probs, cnt_logits, reg_values = self.detection([p3, p4, p5, p6, p7])
        # cls_probs, cnt_logit, reg_values each will have a list of features having shape as below.
        # cls_probs : [[B x 81 x H x W], [B x 81 x H x W], ....]
        # cnt_logits: [[B x 1 x H x W], [B x 1 x H x W], ....]
        # reg_values: [[B x 4 x H x W], [B x 4 x H x W], ....]
        
        return cls_probs, cnt_logits, reg_values
    
    
if __name__ == '__main__':
    import torch
    x = torch.randn(1, 3, 800, 1024)
    model = FCOS()   
    cls_probs, cnt_logits, reg_values = model(x)
    
    for a, b, c in zip(cls_probs, cnt_logits, reg_values):
        print(a.shape, "--", b.shape, "--", c.shape)
        
    