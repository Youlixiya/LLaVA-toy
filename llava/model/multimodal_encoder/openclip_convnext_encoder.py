import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
# from tokenize_anything import model_registry
from .openclip_convnext import CLIPConvNeXt
class BaseProcessor:
    def __init__(self):
        self.transform = lambda x: x
        return

    def __call__(self, item):
        return self.transform(item)

class BaseImageProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)

class OpenCLIPImageProcessor(BaseImageProcessor):
    def __init__(self, image_size=320, mean=None, std=None):
        super().__init__(mean=mean, std=std)
        self.crop_size = {'height':image_size,
                          'width': image_size}
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, images, return_tensors='pt'):
        return self.preprocess(images, return_tensors)
    
    def preprocess(self, images, return_tensors='pt'):
        images_tensor = []
        if type(images) == list:
            for image in images:
                images_tensor.append(self.transform(image))
            images_tensor = torch.stack(images_tensor)
        else:
            images_tensor = self.transform(images)[None]
        return {'pixel_values': images_tensor}
        

class OpenCLIPVisionTower(nn.Module):
    open_clip_checkpoint='./ckpts/openclip/openclip_convnext_large.pt'
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.args = args
        # self.tap = model_registry[self.tap_model_type](checkpoint=self.tap_checkpoint).eval()
        # self.tap.concept_projector.reset_weights(self.concept_weights)
        # self.tap.text_decoder.reset_cache(max_batch_size=8)
        # self.vision_tower = self.tap.image_encoder

        if not delay_load:
            self.load_model()

    def load_model(self):
        self.image_processor = OpenCLIPImageProcessor()
        self.vision_tower = CLIPConvNeXt(self.open_clip_checkpoint, select_layer=self.select_layer)
        self.vision_tower.requires_grad_(False)
        # self.hidden_size=self.vision_tower.output_dim

        self.is_loaded = True

    # def feature_select(self, image_forward_outs):
    #     image_features = image_forward_outs.hidden_states[self.select_layer]
    #     if self.select_feature == 'patch':
    #         image_features = image_features[:, 1:]
    #     elif self.select_feature == 'cls_patch':
    #         image_features = image_features
    #     else:
    #         raise ValueError(f'Unexpected select feature: {self.select_feature}')
    #     return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
                # image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_features = self.vision_tower(images.to(device=self.device, dtype=self.dtype))
            # image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device
    
    # @property
    # def dtype(self):
    #     return self.vision_tower.patch_embed.proj.weight.data.dtype

    # @property
    # def device(self):
    #     return self.vision_tower.patch_embed.proj.weight.data.device

    @property
    def config(self):
        return self.args
        # if self.is_loaded:
        #     return self.vision_tower.config
        # else:
        #     return self.cfg_only

    @property
    def num_patches_per_side(self):
        return 20

    @property
    def num_patches(self):
        return 400
