import torch.nn as nn
import torchvision.models as pre_models


# Return first n layers of a pretrained model
class model_extractor(nn.Module):
    """
        Initialize the model extractor.
        Parameters:
        - arch (str): The architecture of the pretrained model ('alexnet', 'resnet', 'vgg16', etc.).
        - num_layers (int): The number of layers to extract from the model.
        - fix_weights (bool): If True, freeze the weights of the extracted layers to prevent training.
    """
    def __init__(self, arch, num_layers, fix_weights):
        super(model_extractor, self).__init__()
        # Load the specified pretrained model
        if arch.startswith('alexnet') :
            original_model = pre_models.alexnet(pretrained=True)
        elif arch.startswith('resnet') :
            original_model = pre_models.resnet18(pretrained=True)
        elif arch.startswith('vgg16'):
            original_model = pre_models.vgg16(pretrained=True)
        else :
            raise("Not support on this architecture yet")

        # Extract the first `num_layers` layers from the pretrained model
        self.features = nn.Sequential(*list(original_model.children())[:num_layers])

        # Optionally freeze the weights of the extracted layers
        if fix_weights == True:
            for p in self.features.parameters():
                p.requires_grad = False
        # Store the name of the architecture for reference
        self.modelName = arch

    def forward(self, x):
        f = self.features(x)
        return f
# 

