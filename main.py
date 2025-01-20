import torch
import config as cfg
from imagenet10_dataloader import get_data_loaders
from adv_image import Adv_Gen
from cat_adv_image import Cat_Adv_Gen
from regular_generator import regular_generator
from catted_generator import catted_generator
from pre_model_extractor import model_extractor
#-------------------------------
if __name__ == '__main__':
    # Check if CUDA is available and print the result
    print("CUDA Available: ", torch.cuda.is_available())
    # Set the device to CUDA if available and configured to use, otherwise use CPU
    device = torch.device("cuda" if (cfg.use_cuda and torch.cuda.is_available()) else "cpu")

    # Get the data loaders for training and validation datasets
    train_loader, val_loader = get_data_loaders()

    # Extract features using a pre-trained model
    feature_ext = model_extractor(cfg.pretrained_model_arch, cfg.num_layers_ext, cfg.ext_fixed)

    # Check if concatenated generator is to be used
    if cfg.cat_G:
        if cfg.noise_img:
            # Initialize and load the regular generator with noise images
            reg_generator = regular_generator(cfg.num_layers_ext, cfg.ext_fixed, cfg.G_tagged)
            # Load the pre-trained state of the generator from the file specified in cfg.noise_g_path
            reg_generator.load_state_dict(torch.load(cfg.noise_g_path))
            reg_generator.eval()
            # Initialize the concatenated generator
            generator = catted_generator(cfg.num_layers_ext, cfg.ext_fixed, cfg.G_tagged)
            # Create an instance of Cat_Adv_Gen with both generators
            advGen = Cat_Adv_Gen(device, feature_ext, generator, reg_generator)
        else:
            # Initialize the concatenated generator without noise images
            generator = catted_generator(cfg.num_layers_ext, cfg.ext_fixed, cfg.G_tagged)
            advGen = Cat_Adv_Gen(device, feature_ext, generator, False)
    else:
        # Initialize the regular generator
         # Create adversarial generator with concatenated architecture
        # False parameter indicates no regular generator for noise
        generator = regular_generator(cfg.num_layers_ext, cfg.ext_fixed, cfg.G_tagged)
        # Create an instance of Adv_Gen with the regular generator
        advGen = Adv_Gen(device, feature_ext, generator)


