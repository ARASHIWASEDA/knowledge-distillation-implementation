from timm.models.swin_transformer import SwinTransformer
from .registry import register_method

import torch

_architecture = SwinTransformer
_configs = {
    "swin_tiny_patch4_window7_224": {1: [1, (3136, 96)], 2: [3, (784, 192)],
                                     3: [9, (196, 384)], 4: [11, (49, 768)], -1: [-1, 768]},
    "swin_base_patch4_window7_224": {1: [1, (3136, 128)], 2: [3, (784, 256)],
                                     3: [21, (196, 512)], 4: [23, (49, 1024)], -1: [-1, 1024]},
    "swin_nano_patch4_window7_224": {1: [1, (3136, 64)], 2: [3, (784, 128)],
                                     3: [5, (196, 256)], 4: [7, (49, 512)], -1: [-1, 512]},
    "swin_pico_patch4_window7_224": {1: [1, (3136, 48)], 2: [3, (784, 96)],
                                     3: [5, (196, 192)], 4: [7, (49, 384)], -1: [-1, 384]},
    "swin_small_patch4_window7_224": {1: [1, (3136, 96)], 2: [3, (784, 192)],
                                      3: [21, (196, 384)], 4: [23, (49, 768)], -1: [-1, 768]},
}


@register_method
def forward(self, x, requires_feature=False):
    if requires_feature:
        x, feat = self.forward_features(x, requires_feature)
        x = self.forward_head(x, pre_logits=True)
        feat.append(x)
        x = self.head.fc(x)
        return x, feat
    else:
        x = self.forward_features(x, False)
        x = self.forward_head(x)
        return x


@register_method
def forward_features(self, x, requires_feature):
    feat = []
    x = self.patch_embed(x)
    if self.absolute_pos_embed is not None:
        x = x + self.absolute_pos_embed
    x = self.pos_drop(x)
    for layers in self.layers:
        for layer in layers.blocks:
            x = layer(x)
            feat.append(x)
            if layers.downsample is not None:
                x = layers.downsample(x)
    x = self.norm(x)
    feat.append(x)
    return (x, feat) if requires_feature else x


@register_method
def stage_info(self, stage):
    arch = self.default_cfg['architecture']
    index = _configs[arch][stage][0]
    shape = _configs[arch][stage][1]
    return index, shape


@register_method
def is_cnn_model(self):
    return False
