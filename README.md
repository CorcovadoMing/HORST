# HORST

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
