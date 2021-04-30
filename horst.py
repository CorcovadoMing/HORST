# The MIT License (MIT)

# Copyright (c) 2021 NVIDIA CORPORATION.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch
import torch.nn as nn
import torch.nn.functional as F
import opt_einsum as oe



class SpatialFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(2, 1, 3, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: bchw
        avg_pool = x.mean(1, keepdim=True)
        max_pool = x.max(1, keepdim=True)[0]
        return self.f(torch.cat([avg_pool, max_pool], dim=1))

    
    
class STAtt(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.q_pre_f = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.GroupNorm(1, channels)
        )
        self.q_sa = SpatialFilter()
        
        self.k_pre_f = nn.Sequential(
            nn.Conv2d(2*channels, channels, 3, 1, 1, bias=False),
            nn.GroupNorm(1, channels)
        )
        self.k_sa = SpatialFilter()
        
        self.v_pre_f = nn.Sequential(
            nn.Conv2d(2*channels, channels, 3, 1, 1, bias=False),
            nn.GroupNorm(1, channels)
        )
        self.v_sa = SpatialFilter()
        
        self.v_post_f = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(1, channels),
            nn.SiLU(inplace=True),
        )
        
        self.pos_emb = nn.Embedding(8, 2*channels, max_norm=1, scale_grad_by_freq=True)
        
    
    def forward(self, q, k, v):
        # q: (l, b, c, h, w)
        # k: (s, b, c, h, w)
        # v: (s, b, c, h, w)
    
        # Pre-transform
        q = self.q_pre_f(q.reshape(q.size(0)*q.size(1), *q.shape[2:])).reshape(*q.shape)
        k = self.k_pre_f(k.reshape(k.size(0)*k.size(1), *k.shape[2:])).reshape(k.size(0), k.size(1), k.size(2) // 2, *k.shape[3:])
        with torch.no_grad():
            index = torch.arange(v.size(0), device=v.device)
            pos_emb = self.pos_emb(index).contiguous()
            pos_emb = pos_emb[:, None, :, None, None]
            pos_emb = pos_emb.expand_as(v)
        v = v + pos_emb
        v = self.v_pre_f(v.reshape(v.size(0)*v.size(1), *v.shape[2:])).reshape(v.size(0), v.size(1), v.size(2) // 2, *v.shape[3:])
        
        q_sa = self.q_sa(q.reshape(q.size(0)*q.size(1), *q.shape[2:])).reshape(q.size(0), q.size(1), 1, *q.shape[3:])
        k_sa = self.k_sa(k.reshape(k.size(0)*k.size(1), *k.shape[2:])).reshape(k.size(0), k.size(1), 1, *k.shape[3:])
        
        q_fg = (k_sa * q).mean([-1, -2]) # s, b, c
        s_prob = torch.sigmoid(oe.contract('sbc,sbcd->sbd', q_fg, k.reshape(*k.shape[:3], -1))).reshape(v.size(0), v.size(1), 1, v.size(-2), v.size(-1)).expand_as(v)
        self.s_prob = s_prob
        v = v * s_prob
        
        # Temporal attention
        q = (q * q_sa)
        k = (k * k_sa)
        q = q.reshape(*q.shape[:2], -1) # l, b, chw
        k = k.reshape(*k.shape[:2], -1) # s, b, chw
        t_prob = oe.contract('lbd,sbd->lsb', q, k)
        t_prob = F.softmax(t_prob * (q.size(-1) ** -0.5), dim=1)
        self.t_prob = t_prob
        v = oe.contract('lsb,sbchw->bchw', t_prob, v)
        
        # Post-transform to V
        v = self.v_post_f(v)
        return v
    

class ShortcutSqueezeAndExcitation(nn.Module):
    def __init__(self, in_dim, reduction_ratio=4, **kwargs):
        super().__init__()
        
        self.f = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_dim, int(in_dim//reduction_ratio), 1),
            nn.ReLU(),
            nn.Conv2d(int(in_dim//reduction_ratio), in_dim, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        return x + (x * self.f(x))


class HORSTCell(nn.Module):
    def __init__(self,
        input_channels, hidden_channels, steps = 8, kernel_size = 3, bias = True, stride = 1):

        super().__init__()

        ## Input/output interfaces
        self.input_channels  = input_channels
        self.hidden_channels = hidden_channels
        self.stride = stride
        
        self.steps = steps
        
        if stride > 1:
            self.reducer = nn.MaxPool2d(2)
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.SiLU(inplace=True),
        )
        
        self.out_t = nn.Sequential(
            nn.Conv2d(input_channels + hidden_channels, hidden_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.SiLU(inplace=True),
            ShortcutSqueezeAndExcitation(hidden_channels)
        )
        
        self.mha = STAtt(hidden_channels)
    
        self.agg_f = nn.Sequential(
            nn.Conv3d(hidden_channels, hidden_channels, (1,1,2), padding=0, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.SiLU(inplace=True)
        )
        
        
    def initialize(self, inputs):
        device = inputs.device # "cpu" or "cuda"
        dtype = inputs.dtype
        batch_size, _, height, width = inputs.size()
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.hidden_queue = [torch.zeros(batch_size, self.hidden_channels, height // self.stride, width // self.stride, device = device, dtype = dtype)]
        self.output_queue = [torch.zeros(batch_size, self.hidden_channels, height // self.stride, width // self.stride, device = device, dtype = dtype)]
    
    
    def _new_state(self, x, mask):
        new_state = torch.zeros(self.batch_size, *x.shape[1:], device=x.device, dtype=x.dtype)
        new_state[mask] = x
        return new_state

        
    def forward(self, inputs, first_step = False, mask = None):
        if first_step: self.initialize(inputs) # intialize the states at the first step
            
        if self.stride > 1:
            inputs = self.reducer(inputs)
        
        ht = self.encoder(inputs)
        
        k = torch.stack(self.hidden_queue)[:, mask]
        v = torch.stack(self.output_queue)[:, mask]
        kv = torch.cat([k, v], dim=2)
        q = ht.unsqueeze(0)
                
        att = self.mha(
            q,
            kv,
            kv,
        )
        
        if first_step:
            self.output_queue = [self._new_state(att, mask)]
        else:
            self.output_queue.append(self._new_state(att, mask))
            
        att = att + q[0]
        
        outputs = self.out_t(torch.cat([inputs, att], dim=1))
            
        if first_step:
            self.hidden_queue = [self._new_state(ht, mask)]
        else:
            self.hidden_queue.append(self._new_state(ht, mask))
            
        if len(self.hidden_queue) > self.steps:
            self.hidden_queue = self.hidden_queue[-self.steps:]
        
        if len(self.output_queue) > self.steps:
            self.output_queue = self.output_queue[-self.steps:]
        
        return outputs, ht, self.output_queue[-1][mask]


# =============================================================================================

class HORST(nn.Module):
    def __init__(self,
        input_channels,
        layers_per_block, 
        hidden_channels,
        stride = None,
        cell_params = {'steps': 8}, 
        kernel_size = 3, bias = True):
        
        super().__init__()

        ## Hyperparameters
        self.layers_per_block = layers_per_block
        self.hidden_channels  = hidden_channels
        if stride is None:
            stride = [1] * len(self.hidden_channels)
        else:
            stride = stride

        self.num_blocks = len(layers_per_block)
        assert self.num_blocks == len(hidden_channels), "Invalid number of blocks."


        Cell = lambda in_channels, out_channels, stride: HORSTCell(
                                                                   input_channels = in_channels, 
                                                                   hidden_channels = out_channels,
                                                                   steps = cell_params["steps"],
                                                                   kernel_size = kernel_size, 
                                                                   bias = bias, 
                                                                   stride = stride
                                                                  )

        ## Construction of convolutional tensor-train LSTM network

        # stack the convolutional-LSTM layers with skip connections 
        self.layers = nn.ModuleDict()
        for b in range(self.num_blocks):
            for l in range(layers_per_block[b]):
                # number of input channels to the current layer
                if l > 0: 
                    channels = hidden_channels[b]
                elif l == 0 and b == 0:
                    channels = input_channels
                elif l == 0 and b > 0:
                    channels = hidden_channels[b-1]
                    if self.input_shortcut:
                        channels += input_channels
                        
                lid = "b{}l{}".format(b, l) # layer ID
                self.layers[lid] = Cell(channels, hidden_channels[b], stride[b])
            

    def forward(self, inputs, length = None, input_frames = None, future_frames=0, output_frames = None):
        if input_frames is None:
            input_frames = inputs.size(1)
        
        if output_frames is None:
            output_frames = input_frames

        total_steps = input_frames + future_frames
        outputs = [None] * total_steps

        for t in range(total_steps):
            # input_: 4-th order tensor of size [batch_size, input_channels, height, width]
            if t < input_frames: 
                input_ = inputs[:, t]
            
            # length-aware
            if length is None:
                length = torch.stack([torch.LongTensor([total_steps]) for _ in range(input_.size(0))]).to(input_.device).squeeze(-1)
                
            original_batch_size = input_.size(0)
            length_mask = length>t
            input_ = input_[length_mask]                

            backup_input = input_
            queue = [] # previous outputs for skip connection
            output_ = [] # collect the outputs from different layers
            for b in range(self.num_blocks):
                for l in range(self.layers_per_block[b]):
                    lid = "b{}l{}".format(b, l) # layer ID
                    input_, ht, st = self.layers[lid](input_, first_step = (t == 0), mask = length_mask)
                    
                output_.append(input_)
                queue.append(input_)
                
            outputs[t] = output_[-1]
                               
            if length is not None:
                out = torch.zeros(original_batch_size, *outputs[t].shape[1:], device=outputs[t].device, dtype=outputs[t].dtype)
                out[length_mask] = outputs[t]
                outputs[t] = out

        # return the last output_frames of the outputs
        outputs = outputs[-output_frames:]
        
        # 5-th order tensor of size [batch_size, output_frames, channels, height, width]
        outputs = torch.stack(outputs, dim = 1)

        return outputs
