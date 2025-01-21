#import
import torch.nn as nn
import torchvision.models as pre_models
from module.resnet_block import ResnetBlock
from module.pre_model_extractor import model_extractor
import config as cfg

class regular_generator(nn.Module):

    """Regular Generator with flexible encoder depth and optional feature tagging.
    
    This generator uses a ResNet18-based encoder and a decoder with architecture
    that adapts based on the encoder depth. It can optionally apply a feature tag
    to the encoded representation before decoding.
    """

    def __init__(self,
                 num_encoder_layers,
                 fix_encoder,
                 tagged,
                 ):
                     
        # Initialize parent class (nn.Module)
                     
        super(regular_generator, self).__init__()

        self.encoder = model_extractor('resnet18', num_encoder_layers, fix_encoder)

        self.tagged = tagged
        if num_encoder_layers < 5:
            raise("Not support on this layer yet")
        elif num_encoder_layers == 7:
            decoder_lis = [
                ResnetBlock(256),
                ResnetBlock(256),
                
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ConvTranspose2d(256, 128, kernel_size=1, stride=1, bias=False),
                
                ResnetBlock(128),
                ResnetBlock(128),
                
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ConvTranspose2d(128, 64, kernel_size=1, stride=1, bias=False),
                
                ResnetBlock(64),
                ResnetBlock(64),
                
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False),
                nn.Tanh()
                # state size. image_nc x 224 x 224
            ]
        elif num_encoder_layers == 6:
            decoder_lis = [
                ResnetBlock(128),
                ResnetBlock(128),
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ConvTranspose2d(128, 64, kernel_size=1, stride=1, bias=False),
                ResnetBlock(64),
                ResnetBlock(64),
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False),
                nn.Tanh()
                # state size. image_nc x 224 x 224
            ]
        elif num_encoder_layers == 5:
            decoder_lis = [
                ResnetBlock(64),
                ResnetBlock(64),
                ResnetBlock(64),
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False),
                nn.Tanh()
                # state size. image_nc x 224 x 224
            ]

        self.decoder = nn.Sequential(*decoder_lis)

    def forward(self, x):
    """
    Forward pass through the generator.

    Encodes the input tensor, optionally modifies the encoded features by tagging, 
    and generates the output through the decoder.

    Args:
        x (torch.Tensor): Input tensor (e.g., an image or feature tensor).

    Returns:
        tuple:
            - torch.Tensor: The generated output from the decoder.
            - torch.Tensor: The encoded features of the input tensor.
    """
        x_t = self.encoder(x)
        if self.tagged:

            x_t[:, :, :cfg.tag_size, :cfg.tag_size] = x_t.max()
        out = self.decoder(x_t)
        # Return the generated output and the encoded features
        return out, x_t

