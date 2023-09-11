from packaging import version
import torch
from torch import nn
import ipdb

class PatchNCELoss(nn.Module):
    def __init__(self, opt, N_patches=100,nce_t=1):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
        self.N_patches = N_patches
        self.nce_t=nce_t

    def forward(self, feat_q, feat_k):

        feat_k = feat_k.detach()

        num_patches = feat_q.shape[0]

        dim=feat_q.shape[-1]

        pred_psample=feat_q[:,0,:].reshape(num_patches,-1,dim)  #b*n,1,c
        target_psample=feat_k[:,0,:].reshape(num_patches,-1,dim) # b*n,1,c
        target_nsample=feat_k[:,1:,:].reshape(num_patches,-1,dim) # b*n,8,c

        l_neg = torch.bmm(target_nsample,pred_psample.transpose(-2,-1)) # b*N,8,1
        
        l_pos = torch.bmm(pred_psample,target_psample.transpose(-2,-1))# b*N,1,1  
        
        l_neg=l_neg.reshape(num_patches,-1)   #b*N,8

        l_pos=l_pos.reshape(num_patches,-1)  #b*N,1

        
        out=torch.cat((l_pos, l_neg), dim=1)/self.nce_t # b*N,9
        
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,device=feat_q.device))

        return loss

