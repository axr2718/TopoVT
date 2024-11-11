import torch
import torch.nn as nn
import timm
from .betti_encoder import BettiEncoder

class CrossAttention(nn.Module):
    def __init__(self, dim, nheads=8):
        super().__init__()

        self.dim = dim
        self.nheads = nheads
        self.head_dim = dim // nheads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)

        self.k_b0 = nn.Linear(dim, dim)
        self.v_b0 = nn.Linear(dim, dim)
        self.out_b0 = nn.Linear(dim, dim)
        self.norm_b0 = nn.LayerNorm(dim)

        self.k_b1 = nn.Linear(dim, dim)
        self.v_b1 = nn.Linear(dim, dim)
        self.out_b1 = nn.Linear(dim, dim)
        self.norm_b1 = nn.LayerNorm(dim)

        self.features = nn.Sequential(nn.Linear(dim * 2, dim),
                                      nn.LayerNorm(dim),
                                      nn.GELU())
        
    def attention_block(self, q, k, v):
        batch = q.shape[0]

        q = q.view(batch, -1, self.nheads, self.head_dim).transpose(1, 2)
        k = k.view(batch, -1, self.nheads, self.head_dim).transpose(1, 2)
        v = v.view(batch, -1, self.nheads, self.head_dim).transpose(1, 2)

        attention = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attention = attention.softmax(dim=-1)

        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).reshape(batch, -1, self.dim)

        return out
    
    def forward(self, x, b0, b1):
        q = self.q(x)
        
        k_b0 = self.k_b0(b0)
        v_b0 = self.v_b0(b0)
        out_b0 = self.attention_block(q, k_b0, v_b0)
        out_b0 = self.norm_b0(self.out_b0(out_b0))

        k_b1 = self.k_b1(b1)
        v_b1 = self.v_b1(b1)
        out_b1 = self.attention_block(q, k_b1, v_b1)
        out_b1 = self.norm_b1(self.out_b1(out_b1))

        concatenated = torch.cat([out_b0, out_b1], dim=-1)

        return self.features(concatenated)


class TopoViT(nn.Module):
    def __init__(self, model_name, num_classes, betti_dim=256):
        super().__init__()

        self.vit = timm.create_model(model_name, 
                                     pretrained=True, 
                                     num_classes=0)
        
        self.vit.reset_classifier(num_classes)
        
        vit_dim = self.vit.embed_dim

        self.b0_encoder = BettiEncoder(seq_len=100, d_model=betti_dim)
        self.b1_encoder = BettiEncoder(seq_len=100, d_model=betti_dim)

        self.betti_projection = None
        if betti_dim != vit_dim:
            self.betti_projection = nn.Linear(betti_dim, vit_dim)

        self.cross_attention = CrossAttention(dim=vit_dim)

        # self.norm = nn.LayerNorm(vit_dim)

        # self.classifier = nn.Sequential(nn.Linear(vit_dim, vit_dim),
        #                                 nn.GELU(),
        #                                 nn.Dropout(0.1),
        #                                 nn.Linear(vit_dim, num_classes))
        
    def forward(self, img, betti0, betti1):
        vit_features = self.vit.forward_features(img)

        b0_features = self.b0_encoder(betti0)
        b1_features = self.b1_encoder(betti1)

        if self.betti_projection is not None:
            b0_features = self.betti_projection(b0_features)
            b1_features = self.betti_projection(b1_features)

        #vit_features = self.norm(vit_features)

        attention_out = self.cross_attention(vit_features,
                                             b0_features,
                                             b1_features)
        
        vit_features = vit_features + attention_out

        #x = vit_features.mean(dim=1)

        #return self.classifier(x)

        return self.vit.forward_head(vit_features)
