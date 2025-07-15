import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.lorentz.manifold import CustomLorentz
from lib.geoopt.manifolds.stereographic import PoincareBall

from lib.lorentz.layers import LorentzMLR
from lib.poincare.layers import UnidirectionalPoincareMLR

class PoincareManifold(nn.Module):
    def __init__(self,
                 k=1.0,
                 learn_k=True,
                 embed_dim=256,
                 num_classes=19,
                 clip_r=1.0,
                 enc_type='euclidean',
                 manifold_type='poincare'):
        super(PoincareManifold, self).__init__()
        self.k = k
        self.learn_k = learn_k
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.clip_r = clip_r
        self.enc_type = enc_type
        self.dec_type = manifold_type




        if manifold_type == 'poincare':
            self.dec_manifold = PoincareBall(c=self.k, learnable=self.learn_k)
            self.manifold_decoder = UnidirectionalPoincareMLR(self.embed_dim, self.num_classes, True, self.dec_manifold)

        elif manifold_type == 'lorentz':
            self.dec_manifold = CustomLorentz(k=self.k, learnable=self.learn_k)
            self.manifold_decoder = LorentzMLR(self.dec_manifold, self.embed_dim+1, self.num_classes)
            pass
        else:
            pass

    def check_manifold(self, x):
        if self.enc_type=='euclidean' and self.dec_type=='euclidean':
            pass
        elif self.enc_type=='euclidean' and self.dec_type=='lorentz':
            x_norm = torch.norm(x, dim=-1, keepdim=True)
            x = torch.minimum(torch.ones_like(x_norm), self.clip_r / x_norm) * x  # Clipped HNNs
            x = self.dec_manifold.expmap0(F.pad(x, pad=(1, 0), value=0))

        elif self.enc_type=='euclidean' and self.dec_type=='poincare':
            # x_norm = torch.norm(x, dim=-1, keepdim=True)
            x = self.soft_clip(x, clip_r=self.clip_r, margin=0.1)
            # x = torch.minimum(torch.ones_like(x_norm), self.clip_r / x_norm) * x  # Clipped HNNs
            x = self.dec_manifold.expmap0(x)
        return x

    def soft_clip(self, x, clip_r=1.0, margin=0.1):
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        scale = torch.where(
            x_norm > clip_r,
            (clip_r - margin) / x_norm + margin*clip_r / x_norm**2,
            torch.ones_like(x_norm)
        )
        return x*scale

    def forward(self, x):
        ori_shape = x.shape
        if len(ori_shape) == 4:
            B,_,H,W = ori_shape
            x = x.flatten(2).transpose(1,2)
        x = self.check_manifold(x)
        self.check_collapse(x)
        x = self.manifold_decoder(x)
        if len(ori_shape) == 4:
            x = x.transpose(1,2).reshape(B,-1,H,W)
        return x

    def embed(self, x):
        ori_shape = x.shape
        if len(ori_shape) == 4:
            B, _, H, W = ori_shape
            x = x.flatten(2).transpose(1, 2)
        embed = self.check_manifold(x)
        if len(ori_shape) == 4:
            x = embed.transpose(1,2).reshape(B,-1,H,W)
        return x

    def check_collapse(self, embed):
        max_norm = torch.max(torch.norm(embed, dim=-1))
        min_norm = torch.min(torch.norm(embed, dim=-1))
        norm_std = torch.norm(embed, dim=-1).std()
        with open('check_collapse.txt', 'a') as f:
            print('max norm: {}'.format(max_norm), file=f)
            print('min norm: {}'.format(min_norm), file=f)
            print('norm std: {}'.format(norm_std), file=f)

