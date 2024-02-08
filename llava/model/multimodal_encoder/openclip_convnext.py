import torch
from torch import nn
from timm.models.convnext import convnext_large

class CLIPConvNeXt(nn.Module):
    def __init__(self,
                 ckpt_path='./ckpts/openclip/openclip_convnext_large.pt',
                 select_layer=-2):
        super().__init__()
        self.visual = convnext_large()
        self.visual.head.fc = nn.Identity()
        if ckpt_path:
            self.visual.load_state_dict(torch.load(ckpt_path))
        self.visual.requires_grad_(False)
        self.select_layer = select_layer
        self.hidden_size = self.visual.stages[select_layer].blocks[0].norm.weight.shape[0]
    
    @property
    def device(self):
        return self.visual.stem[0].weight.data.device
    
    @property
    def dtype(self):
        return self.visual.stem[0].weight.data.dtype
    
    def forward(self, x, return_all_stage=False):
        x = self.visual.stem(x)
        features = []
        for stage in self.visual.stages:
            x = stage(x)
            features.append(x)
        feature = features[self.select_layer]
        b, c, h, w = feature.shape
        if return_all_stage:
            return features, feature.reshape(b, c, -1).permute(0, 2, 1)
        else:
            return feature.reshape(b, c, -1).permute(0, 2, 1)