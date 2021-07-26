
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import pad



class CustomPadTensor(nn.Module):
    def __init__(self, k_size=(3, 3), stride=1, dilation=1, padding='same'):
        super(CustomPadTensor, self).__init__()
        self.k_size = k_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        
    def forward_old(self, x):
        # Note : This is not perfect when stride=2 and input_dims not divisible by 2.
        #        Resultant shape is not same as TF Conv's SAME padding operation.
        # Taken from : https://github.com/pytorch/pytorch/issues/3867#issuecomment-458423010
        
        if str(self.padding).upper() == 'SAME':
            input_rows, input_cols = [int(x) for x in x.shape[2:4]]     
            # x.shape returns pytorch tensor rather than python int list
            # And doing further computation based on that will grow the graph with such nodes
            # Which needs to be avoided when converting the model to onnx or torchscript.
            filter_rows, filter_cols = self.k_size
        
            out_rows = (input_rows + self.stride - 1) // self.stride
            out_cols = (input_cols + self.stride - 1) // self.stride
        
            padding_rows = max(0, (out_rows - 1) * self.stride +
                                (filter_rows - 1) * self.dilation + 1 - input_rows)
            rows_odd = (padding_rows % 2 != 0)
        
            padding_cols = max(0, (out_cols - 1) * self.stride +
                                (filter_cols - 1) * self.dilation + 1 - input_cols)
            cols_odd = (padding_rows % 2 != 0)
            
            x = pad(x, [padding_cols // 2, (padding_cols // 2) + int(cols_odd),
                        padding_rows // 2, (padding_rows // 2) + int(rows_odd)])        # This is only true for NCHW
                                                                                        # First 2 elements are for last dims
                                                                                        # Next 2 elements are for second last dims
            # Or alternatively we can do as below.
            #x = nn.ZeroPad2d((padding_cols // 2, (padding_cols // 2) + int(cols_odd),
            #            padding_rows // 2, (padding_rows // 2) + int(rows_odd)))(x)

            return x
        else:
            return x
        
    def forward(self, x):
        # Taken from : https://mmuratarat.github.io/2019-01-17/implementing-padding-schemes-of-tensorflow-in-python
        
        if str(self.padding).upper() == 'SAME':
            input_rows, input_cols = [int(x) for x in x.shape[2:4]]     
            # x.shape returns pytorch tensor rather than python int list
            # And doing further computation based on that will grow the graph with such nodes
            # Which needs to be avoided when converting the model to onnx or torchscript.
            filter_rows, filter_cols = self.k_size
        
            if input_rows % self.stride == 0:
                pad_along_height = max((filter_rows - self.stride), 0)
            else:
                pad_along_height = max(filter_rows - (input_rows % self.stride), 0)
            if input_cols % self.stride == 0:
                pad_along_width = max((filter_cols - self.stride), 0)
            else:
                pad_along_width = max(filter_cols - (input_cols % self.stride), 0)
                
            pad_top = pad_along_height // 2 #amount of zero padding on the top
            pad_bottom = pad_along_height - pad_top # amount of zero padding on the bottom
            pad_left = pad_along_width // 2             # amount of zero padding on the left
            pad_right = pad_along_width - pad_left      # amount of zero padding on the right
            
            x = pad(x, [pad_left, pad_right, pad_top, pad_bottom])          # This is only true for NCHW
                                                                            # First 2 elements are for last dim which is Width
                                                                            # Next 2 elements are for second last dim which is Height
            # Or alternatively we can do as below.
            #x = nn.ZeroPad2d((padding_cols // 2, (padding_cols // 2) + int(cols_odd),
            #            padding_rows // 2, (padding_rows // 2) + int(rows_odd)))(x)

            return x
        else:
            return x


class CustomUpsample2d(nn.Module):
    def __init__(self):
        super(CustomUpsample2d, self).__init__()
        
    def forward(self, input, target):
        target_h, target_w = target.size()[2:]
        res = F.interpolate(input, (target_h, target_w), mode='bilinear')    
        return res
        