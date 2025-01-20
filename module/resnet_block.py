# import torch.nn as nn

# # Define a resnet block
# # modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
# class ResnetBlock(nn.Module):
#     def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
#         super(ResnetBlock, self).__init__()
#         self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

#     def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
#         conv_block = []
#         p = 0
#         if padding_type == 'reflect':
#             conv_block += [nn.ReflectionPad2d(1)]
#         elif padding_type == 'replicate':
#             conv_block += [nn.ReplicationPad2d(1)]
#         elif padding_type == 'zero':
#             p = 1
#         else:
#             raise NotImplementedError('padding [%s] is not implemented' % padding_type)

#         conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
#                        norm_layer(dim),
#                        nn.ReLU(True)]
#         if use_dropout:
#             conv_block += [nn.Dropout(0.5)]

#         p = 0
#         if padding_type == 'reflect':
#             conv_block += [nn.ReflectionPad2d(1)]
#         elif padding_type == 'replicate':
#             conv_block += [nn.ReplicationPad2d(1)]
#         elif padding_type == 'zero':
#             p = 1
#         else:
#             raise NotImplementedError('padding [%s] is not implemented' % padding_type)

#         conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
#                        norm_layer(dim)]

#         return nn.Sequential(*conv_block)

import torch.nn as nn

class ResnetBlock(nn.Module):
    """Implements a residual block as used in ResNet architectures.
    
    This block maintains the input dimensions while applying convolutions with residual
    connections. It supports different padding types, normalization, and dropout options.
    
    Modified from CycleGAN implementation: 
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, 
                 use_dropout=False, use_bias=False):
        """Initialize the ResNet block.
        
        Args:
            dim (int): Number of channels in input and output
            padding_type (str): Type of padding ('reflect', 'replicate', or 'zero')
            norm_layer (nn.Module): Normalization layer to use (default: BatchNorm2d)
            use_dropout (bool): Whether to include dropout layer
            use_bias (bool): Whether to include bias in conv layers
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, 
                                              use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct the convolutional block within the ResNet block.
        
        The block consists of:
        Conv1 -> Norm1 -> ReLU -> (Dropout) -> Conv2 -> Norm2
        
        Args:
            dim (int): Number of input/output channels
            padding_type (str): Type of padding to use
            norm_layer (nn.Module): Normalization layer
            use_dropout (bool): Whether to use dropout
            use_bias (bool): Whether to use bias in convolutions
            
        Returns:
            nn.Sequential: The constructed conv block
            
        Raises:
            NotImplementedError: If padding_type is not supported
        """
        conv_block = []
        p = 0  # Padding size
        
        # First conv layer padding
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1  # Use padding in conv layer instead
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
            
        # First conv block: Conv -> Norm -> ReLU
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True)
        ]
        
        # Optional dropout
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
            
        # Reset padding for second conv layer
        p = 0
        # Second conv layer padding
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
            
        # Second conv block: Conv -> Norm
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim)
        ]
        
        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward pass of the ResNet block.
        
        Applies the conv block to input and adds a residual connection.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor after residual connection
        """
        # Residual connection: output = input + conv_block(input)
        out = x + self.conv_block(x) #残差连接
        return out

#     def forward(self, x):
#         out = x + self.conv_block(x)
#         return out
