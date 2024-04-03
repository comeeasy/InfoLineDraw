import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import models
from torchvision.utils import save_image

import lightning.pytorch as pl

import networks
import clip

from pathlib import Path

# from utils import ReplayBuffer
from utils import LambdaLR, createNRandompatches
import networks



class InfoLineDraw(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        
        self.opt = opt
        self.save_hyperparameters(opt)
        
        # Use manual optimization method
        self.automatic_optimization = False
        
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
        self.criterionGAN = networks.GANLoss(use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, reduceme=True)
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionCycleB = torch.nn.L1Loss()
        self.criterionGeom = torch.nn.BCELoss(reduce=True)
        self.criterionCLIP = torch.nn.MSELoss(reduce=True) if not opt.cos_clip else torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        
        # Load pretrained models
        print(f"Load recognition model: InceptionV3(num_classes={opt.num_classes}, mode={opt.mode}, use_aux=True, pretrain=True, freeze=True, every_feat={opt.every_feat==1})")
        self.net_recog = InceptionV3(opt.num_classes, opt.mode=='test', use_aux=True, pretrain=True, freeze=True, every_feat=opt.every_feat==1)
        self.net_recog.eval()
        
        print(f"Load Pretrained CLIP: clip.load('ViT-B/32', device=self.device, jit=False)")
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device, jit=False)
        clip.model.convert_weights(self.clip_model)
        
        
        
    def forward(self, x):
        pass
    
    def training_step(self, batch, batch_idx):
        real_A, real_B, recover_geom, _ = batch['r'], batch['line'], batch['depth'], batch['label']
        
        # Generator A -> B
        fake_B = self.netG_A(real_A)
        rec_A = self.netG_B(fake_B)  # Reconstruction A
        
        # Generator B -> A
        fake_A = self.netG_B(real_B)
        rec_B = self.netG_A(fake_A)  # Reconstruction B

        # losses
        loss_G, loss_D_A, loss_D_B = self.compute_generator_discrimanator_losses(real_A, real_B, fake_A, fake_B, rec_A, rec_B, recover_geom)

        # Update weights
        optimizers= self.optimizers()
        if self.hparams.use_geom and self.hparams.finetune_netGeom:
            optimizer_G_A, optimizer_G_B, optimizer_D_A, optimizer_D_B, optimizer_Geom = optimizers
        else:    
            optimizer_G_A, optimizer_G_B, optimizer_D_A, optimizer_D_B = optimizers
        
        
        optimizer_G_A.zero_grad()
        optimizer_G_B.zero_grad()
        if self.opt.finetune_netGeom == 1:
            optimizer_Geom.zero_grad()
        loss_G.backward()
        optimizer_G_A.step()
        optimizer_G_B.step()
        if self.opt.finetune_netGeom == 1:
            optimizer_Geom.step()
        
        optimizer_D_A.zero_grad()
        loss_D_A.backward()
        optimizer_D_A.step()
        
        optimizer_D_B.zero_grad()
        loss_D_B.backward()
        optimizer_D_B.step()
        
        # Logging
        self.log('train/loss_G', loss_G, batch_size=self.opt.batchSize, prog_bar=True)
        self.log('train/loss_D_A', loss_D_A, batch_size=self.opt.batchSize, prog_bar=True)
        self.log('train/loss_D_B', loss_D_B, batch_size=self.opt.batchSize, prog_bar=True)
        
        if batch_idx % 100 == 0:
            save_image(real_A, os.path.join(self.result_img_dir, f"realA_train_{self.current_epoch}_{batch_idx}.png"))
            save_image(fake_B, os.path.join(self.result_img_dir, f"fakeB_train_{self.current_epoch}_{batch_idx}.png"))
        
        
    def validation_step(self, batch, batch_idx):
        real_A, real_B, recover_geom, _ = batch['r'], batch['line'], batch['depth'], batch['label']
        
        # Generator A -> B
        fake_B = self.netG_A(real_A)
        rec_A = self.netG_B(fake_B)  # Reconstruction A
        
        # Generator B -> A
        fake_A = self.netG_B(real_B)
        rec_B = self.netG_A(fake_A)  # Reconstruction B

        # losses
        loss_G, loss_D_A, loss_D_B = self.compute_generator_discrimanator_losses(real_A, real_B, fake_A, fake_B, rec_A, rec_B, recover_geom)
        
        # Logging
        self.log('val/loss_G', loss_G, batch_size=self.opt.batchSize, prog_bar=True)
        self.log('val/loss_D_A', loss_D_A, batch_size=self.opt.batchSize, prog_bar=True)
        self.log('val/loss_D_B', loss_D_B, batch_size=self.opt.batchSize, prog_bar=True)
        if batch_idx % 10 == 0:
            save_image(real_A, os.path.join(self.result_img_dir, f"realA_val_{self.current_epoch}_{batch_idx}.png"))
            save_image(fake_B, os.path.join(self.result_img_dir, f"fakeB_val_{self.current_epoch}_{batch_idx}.png"))
        
        
        
    def test_step(self, batch, batch_idx):
        real_A, real_B, recover_geom, _ = batch['r'], batch['line'], batch['depth'], batch['label']
        
        # Generator A -> B
        fake_B = self.netG_A(real_A)
        rec_A = self.netG_B(fake_B)  # Reconstruction A
        
        # Generator B -> A
        fake_A = self.netG_B(real_B)
        rec_B = self.netG_A(fake_A)  # Reconstruction B

        # losses
        loss_G, loss_D_A, loss_D_B = self.compute_generator_discrimanator_losses(real_A, real_B, fake_A, fake_B, rec_A, rec_B, recover_geom)
        
        # Logging
        self.log('test/loss_G', loss_G, batch_size=self.opt.batchSize, prog_bar=True)
        self.log('test/loss_D_A', loss_D_A, batch_size=self.opt.batchSize, prog_bar=True)
        self.log('test/loss_D_B', loss_D_B, batch_size=self.opt.batchSize, prog_bar=True)

    # Example helper function for generator losses
    def compute_generator_discrimanator_losses(self, real_A, real_B, fake_A, fake_B, rec_A, rec_B, recover_geom):
        
        if self.opt.use_geom == 1:
            geom_input = fake_B
            if geom_input.size()[1] == 1:
                geom_input = geom_input.repeat(1, 3, 1, 1)
            _, geom_input = self.net_recog(geom_input)

            pred_geom = self.netGeom(geom_input)
            pred_geom = (pred_geom+1)/2.0 ###[-1, 1] ---> [0, 1]

            loss_cycle_Geom = self.criterionGeom(pred_geom, recover_geom)
        else:
            loss_cycle_Geom = 0

        ########## loss A Reconstruction ##########

        loss_G_A = self.criterionGAN(self.netD_A(fake_A), True)

        # GAN loss D_B(G_B(B))
        loss_G_B = self.criterionGAN(self.netD_B(fake_B), True)

        # Forward cycle loss || G_B(G_A(A)) - A||
        loss_cycle_A = self.criterionCycle(rec_A, real_A)
        loss_cycle_B = self.criterionCycleB(rec_B, real_B)
        # combined loss and calculate gradients

        loss_GAN = loss_G_A + loss_G_B
        loss_RC = loss_cycle_A + loss_cycle_B

        loss_G = self.opt.cond_cycle*loss_RC + self.opt.condGAN*loss_GAN
        loss_G += self.opt.condGeom*loss_cycle_Geom


        ### semantic loss
        loss_recog = 0

        # renormalize mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
        recog_real = real_A
        recog_real0 = (recog_real[:, 0, :, :].unsqueeze(1) - 0.48145466) / 0.26862954
        recog_real1 = (recog_real[:, 1, :, :].unsqueeze(1) - 0.4578275) / 0.26130258
        recog_real2 = (recog_real[:, 2, :, :].unsqueeze(1) - 0.40821073) / 0.27577711
        recog_real = torch.cat([recog_real0, recog_real1, recog_real2], dim=1)

        line_input = fake_B
        if self.opt.output_nc == 1:
            line_input_channel0 = (line_input - 0.48145466) / 0.26862954
            line_input_channel1 = (line_input - 0.4578275) / 0.26130258
            line_input_channel2 = (line_input - 0.40821073) / 0.27577711
            line_input = torch.cat([line_input_channel0, line_input_channel1, line_input_channel2], dim=1)

        patches_r = [torch.nn.functional.interpolate(recog_real, size=224)]  #The resize operation on tensor.
        patches_l = [torch.nn.functional.interpolate(line_input, size=224)]

        ## patch based clip loss
        if self.opt.N_patches > 1:
            patches_r2, patches_l2 = createNRandompatches(recog_real, line_input, self.opt.N_patches, self.opt.patch_size)
            patches_r += patches_r2
            patches_l += patches_l2

        loss_recog = 0
        for patchnum in range(len(patches_r)):

            real_patch = patches_r[patchnum]
            line_patch = patches_l[patchnum]

            feats_r = self.clip_model.encode_image(real_patch).detach()
            feats_line = self.clip_model.encode_image(line_patch)

            myloss_recog = self.criterionCLIP(feats_line, feats_r.detach())
            if self.opt.cos_clip == 1:
                myloss_recog = 1.0 - loss_recog
                myloss_recog = torch.mean(loss_recog)

            patch_factor = (1.0 / float(self.opt.N_patches))
            if patchnum == 0:
                patch_factor = 1.0
            loss_recog += patch_factor*myloss_recog
        
        loss_G += self.opt.cond_recog* loss_recog
        
        ##########  Discriminator A ##########

        # Fake loss
        pred_fake_A = self.netD_A(fake_A.detach())
        loss_D_A_fake = self.criterionGAN(pred_fake_A, False)

        # Real loss

        pred_real_A = self.netD_A(real_A)
        loss_D_A_real = self.criterionGAN(pred_real_A, True)

        # Total loss
        loss_D_A = torch.mean(self.opt.condGAN * (loss_D_A_real + loss_D_A_fake) ) * 0.5

        # Fake loss
        pred_fake_B = self.netD_B(fake_B.detach())
        loss_D_B_fake = self.criterionGAN(pred_fake_B, False)

        # Real loss

        pred_real_B = self.netD_B(real_B)
        loss_D_B_real = self.criterionGAN(pred_real_B, True)

        # Total loss
        loss_D_B = torch.mean(self.opt.condGAN * (loss_D_B_real + loss_D_B_fake) ) * 0.5
        
        return loss_G, loss_D_A, loss_D_B

    def configure_optimizers(self):
        optimizer_G_A = optim.Adam(self.netG_A.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        optimizer_G_B = optim.Adam(self.netG_B.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        optimizer_D_A = optim.Adam(self.netD_A.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        optimizer_D_B = optim.Adam(self.netD_B.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        
        lr_scheduler_G_A = torch.optim.lr_scheduler.LambdaLR(optimizer_G_A, lr_lambda=LambdaLR(self.opt.n_epochs, self.opt.epoch, self.opt.decay_epoch).step)
        lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(self.opt.n_epochs, self.opt.epoch, self.opt.decay_epoch).step)
        lr_scheduler_G_B = torch.optim.lr_scheduler.LambdaLR(optimizer_G_B, lr_lambda=LambdaLR(self.opt.n_epochs, self.opt.epoch, self.opt.decay_epoch).step)
        lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(self.opt.n_epochs, self.opt.epoch, self.opt.decay_epoch).step)

        optimizers = [optimizer_G_A, optimizer_G_B, optimizer_D_A, optimizer_D_B]
        lr_schedulers = [lr_scheduler_G_A, lr_scheduler_G_B, lr_scheduler_D_A, lr_scheduler_D_B]
        
        if self.hparams.use_geom and self.hparams.finetune_netGeom:
            optimizer_Geom = optim.Adam(self.netGeom.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
            optimizers.append(optimizer_Geom)
        
        return optimizers, lr_schedulers

    def on_fit_start(self) -> None:
        # Logging images path
        self.result_img_dir = os.path.join(self.logger.log_dir, "result_imgs")
        if not Path(self.result_img_dir).exists():
            Path(self.result_img_dir).mkdir(parents=True, exist_ok=True)
            
        # Checkpoints dir path
        self.checkpoints_path = os.path.join(self.logger.log_dir, "checkpoints")
        if not Path(self.checkpoints_path).exists():
            Path(self.checkpoints_path).mkdir(parents=True, exist_ok=True)
        
    def on_train_epoch_end(self) -> None:
        lr_scheduler_G_A, lr_scheduler_D_B, lr_scheduler_G_B, lr_scheduler_D_A = self.lr_schedulers()
        
        # Update learning rates
        lr_scheduler_G_A.step()
        lr_scheduler_G_B.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        self.opt.checkpoint_dir = self.logger.log_dir
        
        if self.current_epoch % self.opt.save_epoch_freq == 0:
            torch.save(self.netG_A.state_dict(), os.path.join(self.checkpoints_path, f'netG_A_{self.current_epoch}.pth'))
            if self.opt.finetune_netGeom == 1:
                torch.save(self.netGeom.state_dict(), os.path.join(self.checkpoints_path, f'netGeom_{self.current_epoch}.pth'))
            if self.opt.slow == 0:
                torch.save(self.netG_B.state_dict(), os.path.join(self.checkpoints_path, f'netG_B_{self.current_epoch}.pth'))
                torch.save(self.netD_A.state_dict(), os.path.join(self.checkpoints_path, f'netD_A_{self.current_epoch}.pth'))
                torch.save(self.netD_B.state_dict(), os.path.join(self.checkpoints_path, f'netD_B_{self.current_epoch}.pth'))

        torch.save(self.netG_A.state_dict(), os.path.join(self.checkpoints_path, 'netG_A_latest.pth'))
        torch.save(self.netG_B.state_dict(), os.path.join(self.checkpoints_path, 'netG_B_latest.pth'))
        torch.save(self.netD_B.state_dict(), os.path.join(self.checkpoints_path, 'netD_B_latest.pth'))
        torch.save(self.netD_A.state_dict(), os.path.join(self.checkpoints_path, 'netD_A_latest.pth'))
        if self.opt.finetune_netGeom == 1:
            torch.save(self.netGeom.state_dict(), os.path.join(self.checkpoints_path, 'netGeom_latest.pth'))





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