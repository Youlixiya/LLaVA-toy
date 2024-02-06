import os
from .clip_encoder import CLIPVisionTower
from .sam_encoder import SAMVisionTower
from .tap_encoder import TAPVisionTower
from .tinyclip_encoder import TinyCLIPVisionTower
from .openclip_convnext_encoder import OpenCLIPVisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    if 'sam' in vision_tower:
        return SAMVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif 'tap' in vision_tower:
        return TAPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif 'tinyclip' in vision_tower:
        return TinyCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif 'openclip' in vision_tower:
        return OpenCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    is_absolute_path_exists = os.path.exists(vision_tower)
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
