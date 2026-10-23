"""Small RASR-style coarse visual branch, adapted from hidden states to NPU KV.

All image positions enter together. Latent V/Z are NOT supervised embeddings.
No target-model forward/gradient or consistency refinement is implemented.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

def positions(h,w,dim,device):
    y,x=torch.meshgrid(torch.arange(h,device=device),torch.arange(w,device=device),indexing='ij')
    freq=torch.exp(-math.log(10000)*torch.arange(dim//4,device=device)/(dim//4))
    return torch.cat([(x.flatten()[:,None]*freq).sin(),(x.flatten()[:,None]*freq).cos(),
                      (y.flatten()[:,None]*freq).sin(),(y.flatten()[:,None]*freq).cos()],-1)[None]

class Residual(nn.Module):
    def __init__(self,c):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.GELU(),nn.Conv2d(c,c,3,padding=1))
    def forward(self,x):return x+.2*self.net(x)

class Reconstructor(nn.Module):
    def __init__(self,input_dim=1024):
        super().__init__()
        self.input_dim=input_dim
        self.input=nn.Linear(input_dim,256)
        self.inverse_llm=nn.TransformerEncoderLayer(256,8,1024,dropout=0,batch_first=True,norm_first=True)
        self.latent_v=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,1536))
        self.inverse_aligner=nn.Sequential(nn.Linear(1536,768),nn.GELU(),nn.Linear(768,768))
        self.vision_input=nn.Linear(768,192)
        self.inverse_vision=nn.TransformerEncoderLayer(192,6,768,dropout=0,batch_first=True,norm_first=True)
        self.grid_in=nn.Sequential(nn.Conv2d(192,64,1),nn.GELU(),Residual(64))
        self.down=nn.Sequential(nn.Conv2d(64,96,3,stride=2,padding=1),nn.GELU(),Residual(96))
        self.merge=nn.Sequential(nn.Conv2d(160,64,3,padding=1),nn.GELU(),Residual(64))
        layers=[];c=64
        for scale,width in [(3,48),(2,24),(2,12),(2,8)]:
            layers += [nn.Conv2d(c,width*scale*scale,3,padding=1),nn.PixelShuffle(scale),nn.GELU(),Residual(width)]
            c=width
        self.render=nn.Sequential(*layers,nn.Conv2d(c,3,3,padding=1))
        self.coarse=nn.Conv2d(64,3,1)
        nn.init.zeros_(self.coarse.weight);nn.init.zeros_(self.coarse.bias)

    def forward(self,x,grid):
        h,w=grid;b,n,_=x.shape;assert n==h*w
        z=self.input(x)+positions(h,w,256,x.device)
        mask=torch.ones(n,n,device=x.device,dtype=torch.bool).triu(1)
        z=self.inverse_llm(z,src_mask=mask)
        z=self.vision_input(self.inverse_aligner(self.latent_v(z)))+positions(h,w,192,x.device)
        z=self.inverse_vision(z).transpose(1,2).reshape(b,192,h,w)
        skip=self.grid_in(z);small=self.down(skip)
        z=self.merge(torch.cat([skip,F.interpolate(small,size=(h,w),mode='bilinear',align_corners=False)],1))
        return torch.sigmoid(self.render(z)+F.interpolate(self.coarse(z),scale_factor=24,mode='bilinear',align_corners=False))
