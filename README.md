# knowledge-distillation-implementation
这是一个基于pytorch和timm的，针对图像分类网络的，知识蒸馏算法实现教程，支持多种Hint-based和Logits-based的知识蒸馏算法的简单复现。

### 环境准备

以下环境仅供参考：

- NVIDIA CUDA 12.3

- Python 3.10.14

```cmd
pip install -r requirements.txt
```

### 算法支持情况

目前支持的Hint-based的网络有：

- ConvNeXt-Tiny/Small/Base/Large/XLarge
- MobileNetV2-100
- ResNet-18/34/50/101
- vit-tiny/small/base-patch16-224

Logits-based支持timm库中包括的大多数图像分类算法。

已实现的KD算法有：

- CRD
- DIST
- DKD
- NKD
- FITNET
- KD
- OFAKD

上述KD算法的原理简介和具体的实现步骤可以参考我的博客：https://arashiwaseda.github.io/

### 自定义算法和模型

如果想要修改KD算法，请参照以下步骤，关于函数内的实现细节可以参考其他实现的KD算法：

```python
# 1. 首先需要实现一个distiller类，代码格式可以如下
@register_distiller
class YourDistiller(BaseDistiller):
    requires_feature=True  # or False
    def __ini__(self,student,teacher,criterion,args,**kwargs):
        super().__init__()
        """
        Your init code here
        """
    def forward(self,x,y):
        pass
    	"""
    	Your forward code here
    	"""
        return logits_student, losses_dict
# 2. 在distillers目录下的__init__.py文件中导入你的算法
from .your_distiller_file_name import YourDistiller
```

如果想修改支持Hint-based的模型，请参照以下步骤：

```python
"""
首先你需要了解timm中该模型的实现步骤，然后找到forward_feature和forward函数，以ResNet举例
"""
from timm.models.resnet import ResNet
from .registry import register_method

# 修改forward_features方法，使其可以返回特征
@register_method
def forward_features(self, x, requires_feature):
    feat = []
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.act1(x)
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

# 修改forward方法，使其可以返回特征
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
```

