import torch
from torch import nn



class Unroller(nn.Module):
    def __init__(self, in_dim, num_a):
        super().__init__()
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.unrolling = nn.LSTM(in_dim, in_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, num_a)
        )
        
        self.fuse_ev = nn.Sequential(
            nn.Linear(2*in_dim, 2*in_dim),
            nn.BatchNorm1d(2*in_dim),
        )
    
    def forward(self, x, index, n, v):
        x = self.avgpool(x).contiguous()
        x = x[:, None, :].contiguous().repeat(1, 14-index, 1).contiguous()
        
        h0, c0 = self.fuse_ev(torch.cat([n, v], dim=-1)).chunk(2, dim=-1)
        h0 = h0[None, ...].contiguous()
        c0 = c0[None, ...].contiguous()
        
        x = self.unrolling(x, (h0, c0))[0][:, -1, :]
        return self.classifier(x)
        

class Classifier(nn.Module):
    def __init__(self, in_dim, num_a, num_n, num_v):
        super().__init__()
        self.c_a = Unroller(in_dim, num_a)
        
        self.encoder_n = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
        )
        
        self.encoder_v = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
        )
        
        self.c_n = nn.Linear(in_dim, num_n)
        self.c_v = nn.Linear(in_dim, num_v)
        
        self.c_a2n = nn.Linear(num_a, in_dim)
        self.c_a2v = nn.Linear(num_a, in_dim)
    
    def forward(self, x):
        # x: batch, steps, C, H, W, 3
        out = []
        for i in range(x.size(1)):
            a, _, v = x[:, i].split(1, dim=-1)
            e_n = self.encoder_n(v.squeeze(-1))
            e_v = self.encoder_v(v.squeeze(-1))
            c_a = self.c_a(a.squeeze(-1), i, e_n, e_v)
            
            c_n = self.c_n(e_n + self.c_a2n(c_a))
            c_v = self.c_v(e_v + self.c_a2v(c_a))
            out.append((c_a, c_n, c_v))
        return out