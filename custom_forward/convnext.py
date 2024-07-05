from timm.models.convnext import ConvNeXt
from .registry import register_method

_architecture = ConvNeXt
_configs = {
    "convnext_tiny": {1: [0, (96, 56, 56)], 2: [1, (192, 28, 28)],
                      3: [2, (384, 14, 14)], 4: [3, (768, 7, 7)], -1: [-1, 768]},
    "convnext_small": {1: [0, (96, 56, 56)], 2: [1, (192, 28, 28)],
                       3: [2, (384, 14, 14)], 4: [3, (768, 7, 7)], -1: [-1, 768]},
    "convnext_base": {1: [0, (128, 56, 56)], 2: [1, (256, 28, 28)],
                      3: [2, (512, 14, 14)], 4: [3, (1024, 7, 7)], -1: [-1, 1024]},
    "convnext_large": {1: [0, (192, 56, 56)], 2: [1, (384, 28, 28)],
                       3: [2, (768, 14, 14)], 4: [3, (1536, 7, 7)], -1: [-1, 1536]},
    "convnext_xlarge": {1: [0, (256, 56, 56)], 2: [1, (512, 28, 28)],
                        3: [2, (1024, 14, 14)], 4: [3, (2048, 7, 7)], -1: [-1, 2048]},
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
    x = self.stem()
    if requires_feature:
        for stage in self.stages:
            x = stage(x)
            feat.append(x)
        else:
            x = self.stages(x)
        x = self.norm_pre(x)
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
