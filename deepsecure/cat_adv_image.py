# import torch.nn as nn
# import torch
# import torch.nn.functional as F
# import torchvision
# import os
# import config as cfg

# models_path = cfg.models_path
# adv_img_path = cfg.adv_img_path

# # custom weights initialization called on netG and netD
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)


# class Cat_Adv_Gen:
#     def __init__(self,
#                  device,
#                  model_extractor,
#                  generator,
#                  reg_g):

#         self.device = device
#         self.model_extractor = model_extractor
#         self.generator = generator
#         self.box_min = cfg.BOX_MIN
#         self.box_max = cfg.BOX_MAX
#         self.ite = 0
#         #self.CELoss = nn.CrossEntropyLoss()

#         self.model_extractor.to(device)
#         #self.model_extractor.eval()

#         self.generator.to(device)

#         self.noise_generator = reg_g
#         if self.noise_generator != False:
#             self.noise_generator.to(device)

#         # initialize optimizers
#         self.optimizer_G = torch.optim.Adam(self.generator.parameters(),
#                                             lr=0.001)

#         if not os.path.exists(models_path):
#             os.makedirs(models_path)
#         if not os.path.exists(adv_img_path):
#             os.makedirs(adv_img_path)

#     def train_batch(self, x):
#         self.optimizer_G.zero_grad()
#         idx = torch.randint(x.size(0),(x.size(0),))
#         if self.noise_generator != False:
#             with torch.no_grad():
#                 noise_imgs, temp_feature = self.noise_generator(x)
#             adv_imgs, hide_feature = self.generator(x, noise_imgs[idx])
#         else:
#             adv_imgs, hide_feature = self.generator(x, x[idx])
#         adv_img_feature = self.model_extractor(adv_imgs)

#         loss_img = F.l1_loss(adv_imgs, x+5*noise_imgs)
#         loss_adv = F.l1_loss(hide_feature, adv_img_feature)
#         loss_G = loss_img + 1*loss_adv
#         loss_G.backward(retain_graph=True)

#         self.optimizer_G.step()

#         return loss_adv.item(), adv_imgs, idx, loss_img.item()

#     def train(self, train_dataloader, epochs):
#         for epoch in range(1, epochs+1):

#             if epoch == 200:
#                 self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
#                                                     lr=0.0001)
#             if epoch == 400:
#                 self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
#                                                     lr=0.00001)
#             loss_adv_sum = 0
#             loss_img_sum = 0
#             self.ite = epoch
#             for i, data in enumerate(train_dataloader, start=0):
#                 images, labels = data
#                 images, labels = images.to(self.device), labels.to(self.device)

#                 loss_adv_batch, adv_img, idx, loss_img_batch = self.train_batch(images)
#                 loss_adv_sum += loss_adv_batch
#                 loss_img_sum += loss_img_batch


#             # print statistics

#             torchvision.utils.save_image(torch.cat((adv_img[:7], images[:7], (images[idx])[:7])),
#                                          adv_img_path + str(epoch) + ".png",
#                                          normalize=True, scale_each=True, nrow=7)
#             num_batch = len(train_dataloader)
#             print("epoch %d:\n loss_adv: %.3f, loss_img: %.3f \n" %
#                   (epoch, loss_adv_sum/num_batch, loss_img_sum/num_batch))
#             # save generator
#             if epoch%20==0:
#                 netG_file_name = models_path + 'catted_netG_epoch_' + str(epoch) + '.pth'
#                 torch.save(self.generator.state_dict(), netG_file_name)

#             print("check")

import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
import os
import config as cfg

# Get paths from config for saving models and adversarial images
models_path = cfg.models_path
adv_img_path = cfg.adv_img_path

def weights_init(m):
    """Custom weight initialization for the generator and discriminator networks.
    
    Args:
        m: Module whose weights need to be initialized
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # Initialize convolutional layer weights from normal distribution
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        # Initialize batch normalization layer weights and biases
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Cat_Adv_Gen:
    """Concatenated Adversarial Generator class that combines regular and noise generators."""
    
    def __init__(self, device, model_extractor, generator, reg_g):
        """Initialize the Cat_Adv_Gen with necessary components.
        
        Args:
            device: Device to run the model on (CPU/GPU)
            model_extractor: Feature extraction model
            generator: Main generator model
            reg_g: Regular generator for noise generation (False if not used)
        """
        # Store the device (CPU/GPU) on which the model will operate
        self.device = device
        # Assign the feature extraction model for processing input features
        self.model_extractor = model_extractor
        self.generator = generator
        self.box_min = cfg.BOX_MIN  # Minimum pixel value
        self.box_max = cfg.BOX_MAX  # Maximum pixel value
        self.ite = 0  # Iteration counter
        
        # Move models to specified device
        self.model_extractor.to(device)
        self.generator.to(device)
        
        # Setup noise generator if provided
        self.noise_generator = reg_g
        if self.noise_generator != False:
            self.noise_generator.to(device)
            
        # Initialize Adam optimizer for generator
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(),
                                          lr=0.001)
                                          
        # Create directories for saving models and images if they don't exist
        if not os.path.exists(models_path):
            os.makedirs(models_path)
        if not os.path.exists(adv_img_path):
            os.makedirs(adv_img_path)

    def train_batch(self, x):
        """Train generator on a single batch of images.
        
        Args:
            x: Batch of input images
            
        Returns:
            tuple: (adversarial loss, generated images, indices, image loss)
        """
        # Zero out gradients for generator
        self.optimizer_G.zero_grad()
        
        # Generate random indices for image pairing
        idx = torch.randint(x.size(0), (x.size(0),))
        
        if self.noise_generator != False:
            # Generate noise images if noise generator is available
            with torch.no_grad():
                noise_imgs, temp_feature = self.noise_generator(x)
            # Generate adversarial images using input and noise
            adv_imgs, hide_feature = self.generator(x, noise_imgs[idx])
        else:
            # Generate adversarial images using paired original images
            adv_imgs, hide_feature = self.generator(x, x[idx])
            
        # Extract features from adversarial images
        adv_img_feature = self.model_extractor(adv_imgs)
        
        # Calculate losses
        loss_img = F.l1_loss(adv_imgs, x + 5*noise_imgs)  # Image reconstruction loss
        loss_adv = F.l1_loss(hide_feature, adv_img_feature)  # Feature matching loss
        loss_G = loss_img + 1*loss_adv  # Combined generator loss
        
        # Backpropagate and update generator
        loss_G.backward(retain_graph=True)
        self.optimizer_G.step()
        
        return loss_adv.item(), adv_imgs, idx, loss_img.item()

    def train(self, train_dataloader, epochs):
        """Train the generator for specified number of epochs.
        
        Args:
            train_dataloader: DataLoader containing training images
            epochs: Number of epochs to train for
        """
        for epoch in range(1, epochs+1):
            # Reduce learning rate at specific epochs
            if epoch == 200:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                  lr=0.0001)
            if epoch == 400:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                  lr=0.00001)
                                                  
            # Initialize loss sums for epoch
            loss_adv_sum = 0
            loss_img_sum = 0
            self.ite = epoch
            
            # Train on batches
            for i, data in enumerate(train_dataloader, start=0):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                loss_adv_batch, adv_img, idx, loss_img_batch = self.train_batch(images)
                loss_adv_sum += loss_adv_batch
                loss_img_sum += loss_img_batch
                
            # Save sample images for visualization
            torchvision.utils.save_image(
                torch.cat((adv_img[:7], images[:7], (images[idx])[:7])),
                adv_img_path + str(epoch) + ".png",
                normalize=True, scale_each=True, nrow=7
            )
            
            # Print epoch statistics
            num_batch = len(train_dataloader)
            print("epoch %d:\n loss_adv: %.3f, loss_img: %.3f \n" %
                  (epoch, loss_adv_sum/num_batch, loss_img_sum/num_batch))
                  
            # Save generator model every 20 epochs
            if epoch % 20 == 0:
                netG_file_name = models_path + 'catted_netG_epoch_' + str(epoch) + '.pth'
                torch.save(self.generator.state_dict(), netG_file_name)
                
            print("check")
