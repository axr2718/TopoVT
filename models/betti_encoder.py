import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
    

class BettiEncoder(nn.Module):
    def __init__(self,
                 seq_len,
                 d_model=256,
                 nhead=8,
                 num_layers=4,
                 dim_feedforward=1024):
        super().__init__()

        self.input_embedding = nn.Linear(1, d_model)

        self.position_encoder = PositionalEncoding(d_model=d_model, max_len=seq_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   batch_first=True,
                                                   dropout=0)
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                                         num_layers=num_layers,
                                                         norm=nn.LayerNorm(d_model))
        
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.input_embedding(x)
        x = self.position_encoder(x)

        encoded = self.transformer_encoder(x)

        return encoded