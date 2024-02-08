import torch
from torch import nn
from timm.models.convnext import ConvNeXtBlock

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    
class SAMLlavaImageEncoder(nn.Module):
    def __init__(self,
                 clip_vision_tower,
                 sam_adapter_ckpts='',
                 out_chans=256,
                 stages=4):
        super().__init__()
        self.clip_vision_tower = clip_vision_tower
        self.adapter = nn.ModuleList()
        kernel_size_list = [4, 2, 1, -2]
        for i in range(stages):
            input_chans = self.visual.stages[i].blocks[0].norm.weight.shape[0]
            stage = []
            kernel_size = kernel_size_list[i]
            if kernel_size < 0:
                kernel_size = -kernel_size
                stage.append(nn.ConvTranspose2d(
                input_chans, out_chans, kernel_size=kernel_size, stride=kernel_size, padding=0
            ))
            else:
                stage.append(nn.Conv2d(
                    input_chans, out_chans, kernel_size=kernel_size, stride=kernel_size, padding=0
                ))
            stage.append(LayerNorm2d(out_chans))
            stage.append(ConvNeXtBlock(out_chans, out_chans))
            self.adapter.append(nn.Sequential(*stage))
        if adapter_ckpts:
            self.adapter.load_state_dict(torch.load(sam_adapter_ckpts))
    
    def forward(self, x):
        vision_tower = self.clip_vision_tower
        x = vision_tower.stem(x)
        features = []
        for i in range(len(vision_tower.stages)):
            stage = vision_tower.stages[i]
            x = stage(x)
            features.append(self.adapter[i](x))
        return sum(features)