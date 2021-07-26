import torch.nn as nn
import torchvision.models as models


class ResNet50(nn.Module):
    def __init__(self, freeze_backend=[False, False, False, False]):
        super(ResNet50, self).__init__()

        resnet = models.resnet50(pretrained=False)      
                
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


    def forward(self, x):
        # x : [B x 3 x 224 x 224]
        
        c2 = self.c2_feats(x)
        # x : [B x 256 x 56 x 56]
        
        c3 = self.c3_feats(c2)
        # x : [B x 512 x 28 x 28]
		
        c4 = self.c4_feats(c3)
        # x : [B x 1024 x 14 x 14]

        c5 = self.c5_feats(c4)
        # x : [B x 2048 x 7 x 7]
        
        return c3, c4, c5
