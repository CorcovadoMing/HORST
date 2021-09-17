import torch
from torch import nn
import torch.nn.functional as F


class SpatialFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(2, 1, 3, 1, 1), nn.Sigmoid())
    
    def forward(self, x):
        # x: (b, c, h, w)
        # Implementation of Eq. (4) 
        avg_pool = x.mean(1, keepdim=True)
        max_pool = x.max(1, keepdim=True)[0]
        return self.f(torch.cat([avg_pool, max_pool], dim=1))

    
class STATT(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.sf_Q = SpatialFilter()
        self.sf_K = SpatialFilter()
        
        self.q_pre_f = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.GroupNorm(1, channels)
        )
        self.k_pre_f = nn.Sequential(
            nn.Conv2d(2*channels, channels, 3, 1, 1, bias=False),
            nn.GroupNorm(1, channels)
        )
        self.v_pre_f = nn.Sequential(
            nn.Conv2d(2*channels, channels, 3, 1, 1, bias=False),
            nn.GroupNorm(1, channels)
        )        
        self.v_post_f = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(1, channels),
            nn.SiLU(inplace=True),
        )
    
    def forward(self, Q, K, V):
        # q: (1, b, c, h, w)
        # k: (s, b, c, h, w) 
        # v: (s, b, c, h, w)
        
        Q = self.q_pre_f(Q.reshape(Q.size(0)*Q.size(1), *Q.shape[2:])).reshape(*Q.shape)
        K = self.k_pre_f(K.reshape(K.size(0)*K.size(1), *K.shape[2:])).reshape(K.size(0), K.size(1), K.size(2) // 2, *K.shape[3:])
        V = self.v_pre_f(V.reshape(V.size(0)*V.size(1), *V.shape[2:])).reshape(V.size(0), V.size(1), V.size(2) // 2, *V.shape[3:])

        f_Q = self.sf_Q(Q.reshape(Q.size(0)*Q.size(1), *Q.shape[2:])) \
                  .reshape(*Q.shape[:2], 1, *Q.shape[3:])
        f_K = self.sf_K(K.reshape(K.size(0)*K.size(1), *K.shape[2:])) \
                  .reshape(*K.shape[:2], 1, *K.shape[3:])
        
        # Spatial branch - Implementation of Eq. (5)
        f_KQ = (f_K * Q).mean([-1, -2]) # \hat{Q} in Eq. (5)
        S = torch.sigmoid(torch.einsum('sbc,sbcd->sbd', f_KQ, K.reshape(*K.shape[:3], -1)))
        S = S.reshape(*V.shape[:2], 1, *V.shape[-2:]).expand_as(V)
        V = V * S # Eq. (7) - spatial part
        
        # Temporal branch - Implementation of Eq. (6)
        Q = (Q * f_Q).reshape(*Q.shape[:2], -1) 
        K = (K * f_K).reshape(*K.shape[:2], -1) 
        T = torch.einsum('lbd,sbd->lsb', Q, K)
        T = F.softmax(T * (Q.size(-1) ** -0.5), dim=1)
        V = torch.einsum('lsb,sbchw->bchw', T, V) # Eq. (7) - temporal part
        return V
    
    
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
        
        self.mha = STATT(hidden_channels)
    
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