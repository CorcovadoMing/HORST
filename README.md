# Higher Order Recurrent Space-Time Transformer (HORST)

This is the official PyTorch implementation of [Higher Order Recurrent Space-Time Transformer](https://arxiv.org/abs/2104.08665) and [Higher-Order Recurrent Network with Space-Time Attention for Video Early Action Recognition]([https://arxiv.org/abs/2104.08665](https://ieeexplore.ieee.org/abstract/document/9897974)).


## HORST Overview
![](horst_overview.png)

## Spatial-Temporal Attention
Left: Proposed Spatial-Temporal Attention. (Temporal branch is in grey area, and spatial branch is in yellow); Right: Attention in (Vaswani et al. 2017).
![](statt_vs_sa.png)

---

## Usage

```python
import torch
from horst import HORST

model = HORST(
    input_channels=256,
    layers_per_block=[2],
    hidden_channels=[512],
    stride=[1],
)

input = torch.randn(1, 8, 256, 56, 56) # (Batch, Timesteps, Channels, Height, Width)
out = model(input)  # (1, 8, 512, 56, 56)
```

## Citations


```bibtex
@inproceedings{tai2022higher,
  title={Higher-order recurrent network with space-time attention for video early action recognition},
  author={Tai, Tsung-Ming and Fiameni, Giuseppe and Lee, Cheng-Kuang and Lanz, Oswald},
  booktitle={2022 IEEE International Conference on Image Processing (ICIP)},
  pages={1631--1635},
  year={2022},
  organization={IEEE}
}
```


```bibtex
@misc{tai2021higher,
      title={Higher Order Recurrent Space-Time Transformer}, 
      author={Tsung-Ming Tai and Giuseppe Fiameni and Cheng-Kuang Lee and Oswald Lanz},
      year={2021},
      eprint={2104.08665},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```



