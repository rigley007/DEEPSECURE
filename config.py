use_cuda = True
image_nc = 3
epochs = 80
batch_size = 64
BOX_MIN = 0

BOX_MAX = 1
pretrained_model_arch = 'resnet18'
num_layers_ext = 5
ext_fixed = True

G_tagged = True
tag_size = 6

cat_G = True
noise_img = True
noise_g_path = './models/netG_epoch_80.pth'

imagenet10_traindir = 'C:/Users\Rui\Pictures/transfer_imgnet_10/train'
imagenet10_valdir = 'C:/Users\Rui\Pictures/transfer_imgnet_10/val'

models_path = './models/'
adv_img_path = './images/0526/adv/'
