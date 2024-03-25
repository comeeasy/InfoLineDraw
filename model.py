import torch.nn as nn
import torch.nn.functional as F
import torch
import functools
from torchvision import models
from torch.autograd import Variable
import numpy as np
import math

import pytorch_lightning as pl
from torch import optim
import networks
import clip

class InfoDraw(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.save_hyperparameters(opt)
        # Models
        
        print(f"Load generator A: Generator(input_nc={opt.input_nc}, output_nc={opt.output_nc}, n_blocks={opt.n_blocks})")
        self.netG_A = Generator(opt.input_nc, opt.output_nc, opt.n_blocks)
        print(f"Load generator B: Generator(input_nc={opt.input_nc}, output_nc={opt.output_nc}, n_blocks={opt.n_blocks})")
        self.netG_B = Generator(opt.output_nc, opt.input_nc, opt.n_blocks)
        if opt.use_geom:
            # Pre defined.
            self.netGeom = GlobalGenerator2(768, opt.geom_nc, n_downsampling=1, n_UPsampling=3)
            self.netGeom.load_state_dict(torch.load(opt.feats2Geom_path))
            print("Loading pretrained features to depth network from %s"%opt.feats2Geom_path)
            
            if not opt.finetune_netGeom:
                print("Do not finetune netGeom while training")
                self.netGeom.eval()
        
        print(f"Load discriminator A: networks.define_D(input_nc: {opt.input_nc}, ndf: {opt.ndf}, netD: {opt.netD}, n_layers_D: {opt.n_layers_D}, norm: {opt.norm}, use_sigmoid=False")
        self.netD_A = networks.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, use_sigmoid=False)
        print(f"Load discriminator B: networks.define_D(input_nc: {opt.output_nc}, ndf: {opt.ndf}, netD: {opt.netD}, n_layers_D: {opt.n_layers_D}, norm: {opt.norm}, use_sigmoid=False")
        self.netD_B = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, use_sigmoid=False)
        
        # Loss functions
        self.criterionGAN = networks.GANLoss(use_lsgan=True)
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionGeom = torch.nn.BCELoss()
        self.criterionCLIP = torch.nn.MSELoss() if not opt.cos_clip else torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        
        # Load pretrained models
        print(f"Load recognition model: InceptionV3(num_classes={opt.num_classes}, mode={opt.mode}, use_aux=True, pretrain=True, freeze=True, every_feat={opt.every_feat==1})")
        self.net_recog = InceptionV3(opt.num_classes, opt.mode=='test', use_aux=True, pretrain=True, freeze=True, every_feat=opt.every_feat==1)
        self.net_recog.eval()
        
        print(f"Load Pretrained CLIP: clip.load('ViT-B/32', device=self.device, jit=False)")
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device, jit=False)
        clip.model.convert_weights(self.clip_model)
        
    def forward(self, x):
        pass
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        img_r, img_depth, real_B = batch['r'], batch['depth'], batch['line']
        
        # Assuming your model has been appropriately defined and moved to the right device
        # e.g., self.netG_A, self.netG_B, self.netD_A, self.netD_B, self.netGeom, etc.
        # Also, ensure loss functions are defined, e.g., self.criterionGAN, self.criterionCycle, etc.

        # Generator A -> B
        fake_B = self.netG_A(img_r)
        rec_A = self.netG_B(fake_B)  # Reconstruction A
        
        # Generator B -> A
        fake_A = self.netG_B(real_B)
        rec_B = self.netG_A(fake_A)  # Reconstruction B

        # Geometry prediction loss (if applicable)
        loss_cycle_Geom = 0
        if self.opt.use_geom:
            # Prepare input for geometry prediction
            geom_input = fake_B if fake_B.size(1) == 3 else fake_B.repeat(1, 3, 1, 1)
            pred_geom = self.netGeom(geom_input)
            pred_geom = (pred_geom + 1) / 2.0  # Normalize to [0, 1] if needed
            loss_cycle_Geom = self.criterionGeom(pred_geom, img_depth)
        
        # Discriminator losses
        loss_D_A, loss_D_B = self.compute_discriminator_losses(real_A=img_r, real_B=real_B, fake_A=fake_A, fake_B=fake_B)
        
        # Generator losses
        loss_RC, loss_GAN, loss_recog = self.compute_generator_losses(img_r, real_B, fake_A, fake_B, rec_A, rec_B)
        
        # Combine losses
        loss_G = self.cond_cycle * loss_RC + \
                 self.condGAN * loss_GAN + \
                 self.opt.condGeom * loss_cycle_Geom + \
                 self.cond_recog * loss_recog
        
        # Logging
        self.log('loss_G', loss_G)
        self.log('loss_D_A', loss_D_A)
        self.log('loss_D_B', loss_D_B)
        
        # Perform different actions based on optimizer index
        if optimizer_idx == 0:
            return loss_G
        elif optimizer_idx == 1:
            return loss_D_A
        elif optimizer_idx == 2:
            return loss_D_B
        # Include conditions for other optimizers if there are more

    # Example helper function for discriminator losses
    def compute_discriminator_losses(self, real_A, real_B, fake_A, fake_B):
        # Discriminator A Loss: Distinguish real A from fake A images
        pred_real_A = self.netD_A(real_A)
        loss_D_A_real = self.criterionGAN(pred_real_A, True)
        
        pred_fake_A = self.netD_A(fake_A.detach())  # Detach to stop gradients to generator
        loss_D_A_fake = self.criterionGAN(pred_fake_A, False)
        
        # Discriminator B Loss: Distinguish real B from fake B images
        pred_real_B = self.netD_B(real_B)
        loss_D_B_real = self.criterionGAN(pred_real_B, True)
        
        pred_fake_B = self.netD_B(fake_B.detach())  # Detach to stop gradients to generator
        loss_D_B_fake = self.criterionGAN(pred_fake_B, False)
        
        # Combine losses for each discriminator
        loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5
        loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
    
        return loss_D_A, loss_D_B

    def compute_geom_loss(self, ...):
        ...

    # Example helper function for generator losses
    def compute_generator_losses(self, real_A, real_B, fake_A, fake_B, rec_A, rec_B):
        # Adversarial loss to fool discriminators
        pred_fake_A = self.netD_A(fake_A)
        loss_G_A = self.criterionGAN(pred_fake_A, True)
        
        pred_fake_B = self.netD_B(fake_B)
        loss_G_B = self.criterionGAN(pred_fake_B, True)
        
        # Cycle-consistency loss
        loss_cycle_A = self.criterionCycle(rec_A, real_A) # * self.hparams.lambda_A
        loss_cycle_B = self.criterionCycle(rec_B, real_B) # * self.hparams.lambda_B
        
        loss_GAN = loss_G_A + loss_G_B
        loss_RC  = loss_cycle_A + loss_cycle_B
        
        # Combine generator losses
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B
        
        # If there are additional losses (e.g., identity loss, perceptual loss), include them here
        
        return loss_G, loss_G_A, loss_G_B, loss_cycle_A, loss_cycle_B

    def configure_optimizers(self):
        optimizer_G_A = optim.Adam(self.netG_A.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        optimizer_G_B = optim.Adam(self.netG_B.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        optimizer_D_A = optim.Adam(self.netD_A.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        optimizer_D_B = optim.Adam(self.netD_B.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        
        if self.hparams.use_geom and self.hparams.finetune_netGeom:
            optimizer_Geom = optim.Adam(self.netGeom.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
            return [optimizer_G_A, optimizer_G_B, optimizer_D_A, optimizer_D_B, optimizer_Geom], []
        else:
            return [optimizer_G_A, optimizer_G_B, optimizer_D_A, optimizer_D_B], []









norm_layer = nn.InstanceNorm2d

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm_layer(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm_layer(in_features)
                        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, sigmoid=True):
        super(Generator, self).__init__()

        # Initial convolution block
        model0 = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    norm_layer(64),
                    nn.ReLU(inplace=True) ]
        self.model0 = nn.Sequential(*model0)

        # Downsampling
        model1 = []
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model1 += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        norm_layer(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2
        self.model1 = nn.Sequential(*model1)

        model2 = []
        # Residual blocks
        for _ in range(n_residual_blocks):
            model2 += [ResidualBlock(in_features)]
        self.model2 = nn.Sequential(*model2)

        # Upsampling
        model3 = []
        out_features = in_features//2
        for _ in range(2):
            model3 += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        norm_layer(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2
        self.model3 = nn.Sequential(*model3)

        # Output layer
        model4 = [  nn.ReflectionPad2d(3),
                        nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model4 += [nn.Sigmoid()]

        self.model4 = nn.Sequential(*model4)

    def forward(self, x, cond=None):
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)

        return out

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class GlobalGenerator2(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect', use_sig=False, n_UPsampling=0):
        assert(n_blocks >= 0)
        super(GlobalGenerator2, self).__init__()        
        activation = nn.ReLU(True)        

        mult = 8
        model = [nn.ReflectionPad2d(4), nn.Conv2d(input_nc, ngf*mult, kernel_size=7, padding=0), norm_layer(ngf*mult), activation]

        ### downsample
        for i in range(n_downsampling):
            model += [nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, kernel_size=4, stride=2, padding=1),
                      norm_layer(ngf * mult // 2), activation]
            mult = mult // 2

        if n_UPsampling <= 0:
            n_UPsampling = n_downsampling

        ### resnet blocks
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        ### upsample         
        for i in range(n_UPsampling):
            next_mult = mult // 2
            if next_mult == 0:
                next_mult = 1
                mult = 1

            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * next_mult), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * next_mult)), activation]
            mult = next_mult

        if use_sig:
            model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Sigmoid()]
        else:      
            model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)
            
    def forward(self, input, cond=None):
        return self.model(input)


class InceptionV3(nn.Module): #avg pool
    def __init__(self, num_classes, isTrain, use_aux=True, pretrain=False, freeze=True, every_feat=False):
        super(InceptionV3, self).__init__()
        """ Inception v3 expects (299,299) sized images for training and has auxiliary output
        """

        self.every_feat = every_feat

        self.model_ft = models.inception_v3(pretrained=pretrain)
        stop = 0
        if freeze and pretrain:
            for child in self.model_ft.children():
                if stop < 17:
                    for param in child.parameters():
                        param.requires_grad = False
                stop += 1

        num_ftrs = self.model_ft.AuxLogits.fc.in_features #768
        self.model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)

        # Handle the primary net
        num_ftrs = self.model_ft.fc.in_features #2048
        self.model_ft.fc = nn.Linear(num_ftrs,num_classes)

        self.model_ft.input_size = 299

        self.isTrain = isTrain
        self.use_aux = use_aux

        if self.isTrain:
            self.model_ft.train()
        else:
            self.model_ft.eval()


    def forward(self, x, cond=None, catch_gates=False):
        # N x 3 x 299 x 299
        x = self.model_ft.Conv2d_1a_3x3(x)

        # N x 32 x 149 x 149
        x = self.model_ft.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.model_ft.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.model_ft.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.model_ft.Conv2d_4a_3x3(x)

        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.model_ft.Mixed_5b(x)
        feat1 = x
        # N x 256 x 35 x 35
        x = self.model_ft.Mixed_5c(x)
        feat11 = x
        # N x 288 x 35 x 35
        x = self.model_ft.Mixed_5d(x)
        feat12 = x
        # N x 288 x 35 x 35
        x = self.model_ft.Mixed_6a(x)
        feat2 = x
        # N x 768 x 17 x 17
        x = self.model_ft.Mixed_6b(x)
        feat21 = x
        # N x 768 x 17 x 17
        x = self.model_ft.Mixed_6c(x)
        feat22 = x
        # N x 768 x 17 x 17
        x = self.model_ft.Mixed_6d(x)
        feat23 = x
        # N x 768 x 17 x 17
        x = self.model_ft.Mixed_6e(x)

        feat3 = x

        # N x 768 x 17 x 17
        aux_defined = self.isTrain and self.use_aux
        if aux_defined:
            aux = self.model_ft.AuxLogits(x)
        else:
            aux = None
        # N x 768 x 17 x 17
        x = self.model_ft.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.model_ft.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.model_ft.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1
        feats = F.dropout(x, training=self.isTrain)
        # N x 2048 x 1 x 1
        x = torch.flatten(feats, 1)
        # N x 2048
        x = self.model_ft.fc(x)
        # N x 1000 (num_classes)

        if self.every_feat:
            # return feat21, feats, x
            return x, feat21

        return x, aux