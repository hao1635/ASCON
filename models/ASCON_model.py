import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
import ipdb
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torchvision.models as models


class ASCONModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        #parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.1, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=str, default='256', help='number of patches per layer')
        parser.add_argument('--layer_weight', type=str, default='1', help='number of weight per layer')
        parser.add_argument('--k_size', type=str, default='3', help='number of k')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G', 'D']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
        self.num_patches = [int(i) for i in self.opt.num_patches.split(',')]
        self.layer_weight = torch.tensor([float(i) for i in self.opt.layer_weight.split(',')])
        self.m = opt.m  #Momentum

        if self.isTrain:
            self.model_names = ['G','Online','Target','Projection_online','Projection_target','Predictor']
        else:  # during test time, only load G
            self.model_names = ['G']

        initialize_weights=True
        if opt.netG == 'ESAU':
            self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt,initialize_weights=False)
        else:
            self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt,initialize_weights=True)

        #ipdb.set_trace()
        if self.isTrain:
            netOnline = networks.Disentangle_UNet(n_channels=1, n_classes=1, bilinear=True)
            self.netOnline = networks.init_net(netOnline,gpu_ids=opt.gpu_ids,initialize_weights=initialize_weights)
            netTarget = networks.Disentangle_UNet(n_channels=1, n_classes=1, bilinear=True)
            self.netTarget = networks.init_net(netTarget,gpu_ids=opt.gpu_ids,initialize_weights=initialize_weights)

            self.netProjection_online = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt,initialize_weights=initialize_weights).to(self.device)
            self.netProjection_target = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt,initialize_weights=initialize_weights).to(self.device)
            self.initializes_target_network()

            self.netPredictor = nn.Sequential(
                nn.Linear(256, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 256)
            ).to(self.device)

            self.initializes_target_network()

        if self.isTrain:
            # define loss functions
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))
 
            self.optimizer_G = torch.optim.AdamW(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)

            self.optimizer_R = torch.optim.SGD(list(self.netOnline.parameters())+list(self.netProjection_online.parameters())+list(self.netPredictor.parameters()),lr=self.opt.lr)
            self.optimizers.append(self.optimizer_R)

        self.SPloss= torch.nn.MSELoss()

        self.phase=opt.phase

    @torch.no_grad()
    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.netOnline.parameters(), self.netTarget.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.netProjection_online.parameters(), self.netProjection_target.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient


    def optimize_parameters(self):
        # forward
        self.forward()

        #update D
        self.set_requires_grad(self.netOnline, True)
        self.set_requires_grad(self.netProjection_online, True)
        self.set_requires_grad(self.netPredictor, True)
        self.optimizer_R.zero_grad()

        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()

        self.optimizer_R.step()
        self._update_target_network_parameters()
            

        #update G
        self.set_requires_grad(self.netOnline, False)
        self.set_requires_grad(self.netProjection_online, False)
        self.set_requires_grad(self.netPredictor, False)

        self.optimizer_G.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()


    def set_input(self, input):

        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG(self.real_A)


    def compute_D_loss(self):

        # Fake; stop backprop to the generator by detaching fake_B
        self.loss_D = self.MAC_Net(self.real_B, self.fake_B.detach())
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        #ipdb.set_trace()

        if self.opt.epoch_count > 0:
            self.loss_NCE = self.MAC_Net(self.real_B, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        self.loss_S = 10*self.SPloss(self.fake_B, self.real_B)+(1-util.compute_ssim(self.fake_B, self.real_B))
        self.loss_G = 0.1*self.loss_NCE + 10*self.loss_S

        return self.loss_G

    def MAC_Net(self, real_B,fake_B):
        n_layers = len(self.nce_layers)

        patch_size=real_B.shape[-1]

        if self.opt.local_variance==True:
            pixweght=self.real_B
        else:
            pixweght=None

        with torch.no_grad():
            feat_k_1 = self.netTarget(real_B,self.nce_layers, encode_only=True)
            feat_k_pool_1, sample_ids, sample_local_ids, sample_top_idxs = self.netProjection_target(patch_size,feat_k_1, self.num_patches,None,None,None,pixweght=None)

        feat_q_1 = self.netOnline(fake_B, self.nce_layers, encode_only=True)
        feat_q_pool_1, _, _ , _ =  self.netProjection_online(patch_size,feat_q_1, self.num_patches, sample_ids,sample_local_ids,sample_top_idxs,pixweght=pixweght)  #online

        total_nce_loss = 0.0
        for i,(f_q_1, f_k_1,crit) in enumerate(zip(feat_q_pool_1,feat_k_pool_1,self.criterionNCE)):
            if i==0:
                loss =  self.regression_loss(self.netPredictor(f_q_1), f_k_1.detach())
            if i==1:
                loss = crit(f_q_1, f_k_1.detach())

            weight=torch.tensor(self.layer_weight[i])/torch.sum(self.layer_weight)

            total_nce_loss += weight*loss.mean()

        return total_nce_loss        

    def compute_metrics(self):
        with torch.no_grad():
            y_pred=self.fake_B
            y=self.real_B

            psnr=util.compute_psnr(y_pred,y)
            ssim=util.compute_ssim(y_pred,y)
            rmse=util.compute_rmse(y_pred,y)
            #ipdb.set_trace()
            if self.phase == 'test':
                return psnr,ssim,rmse

            if 'train' in self.phase :
                return self.loss_D,self.loss_G,psnr,ssim,rmse

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)
    
    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.netOnline.parameters(), self.netTarget.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        for param_q, param_k in zip(self.netProjection_online.parameters(), self.netProjection_target.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        self.vgg = models.vgg19(pretrained=True).features
        self.device = device
        self.criterion = nn.MSELoss()
        self.layer_indices = {'3': 4, '8': 9, '17': 18, '26': 27, '35': 36}

    def forward(self, x, y):
        x_vgg, y_vgg = self.get_features(x), self.get_features(y)
        loss = 0
        for key in x_vgg:
            loss += self.criterion(x_vgg[key], y_vgg[key])
        return loss

    def get_features(self, x):
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layer_indices:
                features[self.layer_indices[name]] = x
        return features.to(self.device)
