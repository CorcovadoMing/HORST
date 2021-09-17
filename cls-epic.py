import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(self, in_dim, num_a, num_n, num_v):
        super().__init__()
        self.c_a = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(3*in_dim, num_a)
        )
        
        self.c_n = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_dim, num_n)
        )
        
        self.c_v = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_dim, num_v)
        )
        
        self.c_a2n = nn.Linear(num_a, num_n)
        self.c_a2v = nn.Linear(num_a, num_v)
    
    def forward(self, x):
        # x: batch, steps, C, H, W, 3
        out = []
        for i in range(x.size(1)):
            a, n, v = x[:, i].split(1, dim=-1) # y_t, e_t, STATT_t 
            c_a = self.c_a(torch.cat([a.squeeze(-1), n.squeeze(-1), v.squeeze(-1)], dim=1))
            c_n = self.c_n(n.squeeze(-1)) + self.c_a2n(c_a)
            c_v = self.c_v(v.squeeze(-1)) + self.c_a2v(c_a)
            out.append((c_a, c_n, c_v))
        return out