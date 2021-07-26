import numpy as np
import torch.nn as nn
import torchvision.models as models
from torch.utils.checkpoint import checkpoint


class ResNet18(nn.Module):
    def __init__(self, freeze_backend=[False, False, False, False], \
                        use_gradient_checkpointing=False):
        super(ResNet18, self).__init__()

        self.use_gradient_checkpointing = use_gradient_checkpointing

        resnet = models.resnet18(pretrained=False)      
        
        self.c2_feats = nn.Sequential()
        for layer_name, layer_module in list(resnet.named_children())[:5]:
            self.c2_feats.add_module(layer_name, layer_module)
        
        self.c3_feats = nn.Sequential()
        for layer_name, layer_module in list(resnet.named_children())[5:6]:
            self.c3_feats.add_module(layer_name, layer_module)

        self.c4_feats = nn.Sequential()
        for layer_name, layer_module in list(resnet.named_children())[6:7]:
            self.c4_feats.add_module(layer_name, layer_module)

        self.c5_feats = nn.Sequential()
        for layer_name, layer_module in list(resnet.named_children())[7:8]:
            self.c5_feats.add_module(layer_name, layer_module)

        for freeze_layer, layer in zip(freeze_backend, [self.c2_feats, self.c3_feats, self.c4_feats, self.c5_feats]):
            if freeze_layer:
                # Fix the parameters of the feature extractor: 
                for param in layer.parameters(): 
                    param.requires_grad = False

        # Modify the batchnorm momentum to account for two times accumulation
        # of batch mean / stddev to running mean / stddev.
        # Two times accumulation happens due to checkpointing only specific tensors.
        # For rest of the tensors their gradients will be computed again during backprop
        # whenever required from checkpointed tensors.
        # During these the batchnorm running stats gets updated again using the same input
        # data. To avoid that we can modify the momentum as
        # new_momentum = 1 - np.sqrt(1 - old_mom)
        # Ref. : https://discuss.pytorch.org/t/checkpoint-with-batchnorm-running-averages/17738/8
        if self.use_gradient_checkpointing:
            for name, module in list(self.named_modules()):
                if isinstance(module, nn.BatchNorm2d):
                    layer_module.momentum = 1 - np.sqrt(1-0.1)
        else:
            # Default value of momentum is used which is 0.1
            pass 


    def forward(self, x):
        # x : [B x 3 x 224 x 224]
        
        if self.use_gradient_checkpointing:
            c2 = checkpoint(self.c2_feats, x)
        else:
            c2 = self.c2_feats(x)
        # x : [B x 256 x 56 x 56]
        
        if self.use_gradient_checkpointing:
            c3 = checkpoint(self.c3_feats, c2)
        else:
            c3 = self.c3_feats(c2)
        # x : [B x 512 x 28 x 28]
		
        if self.use_gradient_checkpointing:
            c4 = checkpoint(self.c4_feats, c3)
        else:
            c4 = self.c4_feats(c3)
        # x : [B x 1024 x 14 x 14]

        if self.use_gradient_checkpointing:
            c5 = checkpoint(self.c5_feats, c4)
        else:
            c5 = self.c5_feats(c4)
        # x : [B x 2048 x 7 x 7]
        
        return c3, c4, c5
