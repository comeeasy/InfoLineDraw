import argparse

import torch
import torchvision
import torchvision.transforms as transforms

from PIL import Image

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

from model import InfoLineDraw
from dataset import UnpairedDepthDataModule



def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='name of this experiment')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='Where checkpoints are saved')
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
    parser.add_argument('--batchSize', type=int, default=4, help='size of the batches')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation', default=True)
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')

    ###loading data
    parser.add_argument('--dataset_root', type=str, default="", help="dataset_root for InfoLineDraw")
    parser.add_argument('--dataroot', type=str, default='datasets/vangogh2photo/', help='photograph directory root directory')
    parser.add_argument('--root2', type=str, default='', help='line drawings dataset root directory')
    parser.add_argument('--depthroot', type=str, default='', help='dataset of corresponding ground truth depth maps')
    parser.add_argument('--feats2Geom_path', type=str, default='checkpoints/feats2Geom/feats2depth.pth', 
                                    help='path to pretrained features to depth map network')

    ### architecture and optimizers
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for optimizer')
    parser.add_argument('--decay_epoch', type=int, default=50, help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
    parser.add_argument('--geom_nc', type=int, default=3, help='number of channels of geom data')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
    parser.add_argument('--netD', type=str, default='basic', help='selects model to use for netD')
    parser.add_argument('--n_blocks', type=int, default=3, help='number of resnet blocks for generator')
    parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
    parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
    parser.add_argument('--disc_sigmoid', type=int, default=0, help='use sigmoid in disc loss')
    parser.add_argument('--every_feat', type=int, default=1, help='use transfer features for recog loss')
    parser.add_argument('--finetune_netGeom', type=int, default=1, help='make geometry networks trainable')

    ### loading from checkpoints
    parser.add_argument('--load_pretrain', type=str, default='', help='where to load file if wanted')
    parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
    parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load from if continue_train')

    ### dataset options
    parser.add_argument('--mode', type=str, default='train', help='train, val, test, etc')
    parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
    parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
    parser.add_argument('--aspect_ratio', type=float, default=1.0, help='The ratio width/height. The final height of the load image will be crop_size/aspect_ratio')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
    parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
    parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')

    ######## loss weights
    parser.add_argument("--cond_cycle", type=float, default=0.1, help="weight of the appearance reconstruction loss")
    parser.add_argument("--condGAN", type=float, default=1.0, help="weight of the adversarial style loss")
    parser.add_argument("--cond_recog", type=float, default=10.0, help="weight of the semantic loss")
    parser.add_argument("--condGeom", type=float, default=10.0, help="weight of the geometry style loss")

    ### geometry loss options
    parser.add_argument('--use_geom', type=int, default=1, help='include the geometry loss')
    parser.add_argument('--midas', type=int, default=1, help='use midas depth map')

    ### semantic loss options
    parser.add_argument('--N_patches', type=int, default=1, help='number of patches for clip')
    parser.add_argument('--patch_size', type=int, default=128, help='patchsize for clip')
    parser.add_argument('--num_classes', type=int, default=55, help='number of classes for inception')
    parser.add_argument('--cos_clip', type=int, default=0, help='use cosine similarity for CLIP semantic loss')

    ### save options
    parser.add_argument('--save_epoch_freq', type=int, default=1000, help='how often to save the latest model in steps')
    parser.add_argument('--slow', type=int, default=0, help='only frequently save netG_A, netGeom')
    parser.add_argument('--log_int', type=int, default=50, help='display frequency for tensorboard')


    opt = parser.parse_args()
    print(opt)

    transforms_r = [
        transforms.Resize(int(opt.size * 1.12), Image.BICUBIC),
        transforms.RandomCrop(opt.size),
        transforms.ToTensor()
    ]
    data_module = UnpairedDepthDataModule(opt, transforms_r, 16)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss_D_B',
        mode="min",
        filename='wallpaper_fault-{epoch:02d}-{val_F1Score:.2f}',
        save_last=True,
        every_n_epochs=10,
    )

    trainer = pl.Trainer(
        max_epochs=100, 
        devices="auto", 
        callbacks=[checkpoint_callback], 
        benchmark=True,
        log_every_n_steps=5,
    )
    
    model = InfoLineDraw(opt)
    
    trainer.fit(model=model, datamodule=data_module)

    trainer.test(model=model, datamodule=data_module)
    



if __name__ == "__main__":
    train()


