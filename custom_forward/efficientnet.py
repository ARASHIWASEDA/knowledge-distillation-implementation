from timm.models.efficientnet import EfficientNet
from .registry import register_method

import torch

_architecture = EfficientNet
_configs = {
    "mobilenetv2_100": {1: [2, (24, 56, 56)], 2: [5, (32, 28, 28)],
                        3: [12, (96, 14, 14)], 4: [17, (1280, 7, 7)], -1: [-1, 1280]}
}


@register_method
def forward(self, x, requires_feature=False):
    if requires_feature:
        x, feat = self.forward_features(x, requires_feature)
        x = self.forward_head(x, pre_logits=True)
        feat.append(x)
        x = self.classifier(x)
        return x, feat
    else:
        x = self.forward_features(x, False)
        x = self.forward_head(x)
        return x


@register_method
def forward_features(self, x, requires_feature):
    feat = []
    x = self.conv_stem(x)
    x = self.bn1(x)
    if requires_feature:
        for blks in self.blocks:
            for blk in blks:
                x = blk(x)
                feat.append(x)
    else:
        x = self.blocks(x)
    x = self.conv_head(x)
    x = self.bn2(x)
    feat.append(x)
    return (x, feat) if requires_feature else x


@register_method
def stage_info(self, stage):
    arch = self.default_cfg['architecture']
    index = _configs[arch][stage][0]
    shape = _configs[arch][stage][1]
    return index, shape


@register_method
def is_cnn_model():
    return True
