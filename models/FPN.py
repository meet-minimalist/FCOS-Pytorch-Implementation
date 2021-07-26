
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from models.model_utils import CustomPadTensor, CustomUpsample2d


class FPN(nn.Module):
    def __init__(self, pyramid_features=[512, 1024, 2048], features=256, \
                    use_gradient_checkpointing=False):
        super(FPN, self).__init__()

        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        self.proj_p5_1x1 = nn.Conv2d(pyramid_features[2], features, kernel_size=1, stride=1)
        self.proj_p4_1x1 = nn.Conv2d(pyramid_features[1], features, kernel_size=1, stride=1)
        self.proj_p3_1x1 = nn.Conv2d(pyramid_features[0], features, kernel_size=1, stride=1)
        
        self.conv_p3_3x3 = nn.Conv2d(features, features, kernel_size=3, stride=1)
        self.conv_p4_3x3 = nn.Conv2d(features, features, kernel_size=3, stride=1)
        self.conv_p5_3x3 = nn.Conv2d(features, features, kernel_size=3, stride=1)
        
        self.conv_p6_3x3 = nn.Conv2d(features, features, kernel_size=3, stride=2)
        self.conv_p7_3x3 = nn.Conv2d(features, features, kernel_size=3, stride=2)
        
        self.relu = nn.ReLU()
        
        self.pad_1x1 = CustomPadTensor(k_size=(1, 1), padding='same')
        self.pad_3x3_s1 = CustomPadTensor(k_size=(3, 3), stride=1, padding='same')
        self.pad_3x3_s2 = CustomPadTensor(k_size=(3, 3), stride=2, padding='same')
        self.upsample = CustomUpsample2d()
            
        
    def forward(self, x):
        c3, c4, c5 = x
        # For Resnet50-backbone
        # c3 : [B x 512 x 100 x 128]
        # c4 : [B x 1024 x 50 x 64]
        # c5 : [B x 2048 x 25 x 32]

        # For Resnet18-backbone
        # c3 : [B x 128 x 100 x 128]
        # c4 : [B x 256 x 50 x 64]
        # c5 : [B x 512 x 25 x 32]
        
        p5_1x1 = self.proj_p5_1x1(self.pad_1x1(c5))
        # p5_1x1 : [B x 256 x 25 x 32] 
        
        p5_pad = self.pad_3x3_s1(p5_1x1)
        if self.use_gradient_checkpointing:
            p5_3x3 = checkpoint(self.conv_p5_3x3, p5_pad)
        else:
            p5_3x3 = self.conv_p5_3x3(p5_pad)
        # p5_3x3 : [B x 256 x 25 x 32] 

        p4_1x1 = self.proj_p4_1x1(self.pad_1x1(c4))
        # p4_1x1 : [B x 256 x 50 x 64]

        p4_pad = self.pad_3x3_s1(p4_1x1 + self.upsample(p5_1x1, p4_1x1))
        if self.use_gradient_checkpointing:
            p4_3x3 = checkpoint(self.conv_p4_3x3, p4_pad)
        else:
            p4_3x3 = self.conv_p4_3x3(p4_pad)
        # p4_3x3 : [B x 256 x 50 x 64]

        p3_1x1 = self.proj_p3_1x1(self.pad_1x1(c3))
        # p3_1x1 : [B x 256 x 100 x 128]

        p3_pad = self.pad_3x3_s1(p3_1x1 + self.upsample(p4_1x1, p3_1x1))
        if self.use_gradient_checkpointing:
            p3_3x3 = checkpoint(self.conv_p3_3x3, p3_pad)
        else:
            p3_3x3 = self.conv_p3_3x3(p3_pad)
        # p3_3x3 : [B x 256 x 100 x 128]

        p6_pad = self.pad_3x3_s2(p5_3x3)
        if self.use_gradient_checkpointing:
            p6_3x3 = checkpoint(self.conv_p6_3x3, p6_pad)
        else:
            p6_3x3 = self.conv_p6_3x3(p6_pad)
        # p6_3x3 : [B x 256 x 13 x 16]

        p7_pad = self.pad_3x3_s2(self.relu(p6_3x3))
        if self.use_gradient_checkpointing:
            p7_3x3 = checkpoint(self.conv_p7_3x3, p7_pad)
        else:
            p7_3x3 = self.conv_p7_3x3(p7_pad)
        # p7_3x3 : [B x 256 x 7 x 8]
         
        # p3_3x3 : [B x 256 x 100 x 128]
        # p4_3x3 : [B x 256 x 50 x 64]
        # p5_3x3 : [B x 256 x 25 x 32] 
        # p6_3x3 : [B x 256 x 13 x 16]
        # p7_3x3 : [B x 256 x 7 x 8]
        return p3_3x3, p4_3x3, p5_3x3, p6_3x3, p7_3x3
