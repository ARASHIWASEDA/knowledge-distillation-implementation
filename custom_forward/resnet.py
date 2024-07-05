from timm.models.resnet import ResNet
from .registry import register_method

_architecture = ResNet
_configs = {
    "resnet18": {1: [0, (64, 56, 56)], 2: [1, (128, 28, 28)],
                 3: [2, (256, 14, 14)], 4: [3, (512, 7, 7)], -1: [-1, 512]},
    "resnet34": {1: [0, (64, 56, 56)], 2: [1, (128, 28, 28)],
                 3: [2, (256, 14, 14)], 4: [3, (512, 7, 7)], -1: [-1, 512]},
    "resnet50": {1: [0, (256, 56, 56)], 2: [1, (512, 28, 28)],
                 3: [2, (1024, 14, 14)], 4: [3, (2048, 7, 7)], -1: [-1, 2048]},
    "resnet101": {1: [0, (256, 56, 56)], 2: [1, (512, 28, 28)],
                  3: [2, (1024, 14, 14)], 4: [3, (2048, 7, 7)], -1: [-1, 2048]}
}


@register_method
def forward_features(self, x, requires_feature):
    feat = []
    x = self.conv1
    x = self.bn1
    x = self.act1
    x = self.maxpool(x)

    x = self.layer1(x)
    feat.append(x)
    x = self.layer2(x)
    feat.append(x)
    x = self.layer3(x)
    feat.append(x)
    x = self.layer4(x)
    feat.append(x)

    return (x, feat) if requires_feature else x


@register_method
def forward(self, x, requires_feature=False):
    if requires_feature:
        x, feat = self.forward_features(x, requires_feature)
        x = self.forward_head(x, pre_logits=True)
        feat.append(x)
        x = self.fc(x)
        return x, feat
    else:
        x = self.forward_features(x, False)
        x = self.forward_head(x)
        return x


@register_method
def stage_info(self, stage):
    arch = self.default_cfg['architecture']
    index = _configs[arch][stage][0]
    shape = _configs[arch][stage][1]
    return index, shape


@register_method
def is_cnn_model():
    return True
