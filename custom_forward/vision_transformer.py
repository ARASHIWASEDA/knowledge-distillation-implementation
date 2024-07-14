from .registry import register_method

import torch
from timm.models.vision_transformer import VisionTransformer

_architecture = VisionTransformer
_configs = {
    "vit_small_patch16_224": {1: [1, (197, 384)], 2: [3, (197, 384)],
                              3: [9, (197, 384)], 4: [11, (197, 384)], -1: [-1, 384]},
    "vit_base_patch16_224": {1: [1, (197, 768)], 2: [3, (197, 768)],
                             3: [9, (197, 768)], 4: [11, (197, 768)], -1: [-1, 768]},
    "vit_tiny_patch16_224": {1: [1, (197, 192)], 2: [3, (197, 192)],
                             3: [9, (197, 192)], 4: [11, (197, 192)], -1: [-1, 192]}
}


@register_method
def forward(self, x, requires_feature=False):
    if requires_feature:
        x, feat = self.forward_features(x, requires_feature)
        x = self.forward_head(x, pre_logits=True)
        feat.append(x)
        x = self.head(x)
        return x, feat
    else:
        x = self.forward_features(x, False)
        x = self.forward_head(x)
        return x


@register_method
def forward_features(self, x, requires_feature):
    feat = []
    x = self.patch_embed(x)
    x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = self.pos_drop(x + self.pos_embed)
    if requires_feature:
        for blk in self.blocks:
            x = blk(x)
            feat.append(x)
    else:
        x = self.blocks(x)
    x = self.norm(x)
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
