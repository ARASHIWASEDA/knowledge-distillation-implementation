import torch.nn as nn
import torch

from timm.models.layers import _assert, trunc_normal_


class SepConv(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.op(x)


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, out_dim=None, norm_layer=nn.LayerNorm, act_layer=nn.Identity):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim or 2 * dim
        self.norm = norm_layer(4 * dim)

        in_features = 4 * dim
        self.reduction = nn.Linear(in_features, self.out_dim, bias=False)
        self.act = act_layer()

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        _assert(L == H * W, "input feature has wrong size")
        _assert(H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even.")

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)
        x = self.act(x)

        return x


class GAP1d(nn.Module):
    def __init__(self):
        super(GAP1d, self).__init__()

    def forward(self, x):
        return x.mean(1)


class TokenFilter(nn.Module):
    """remove cls tokens in forward"""

    def __init__(self, number=1, inverse=False, remove_mode=True):
        super(TokenFilter, self).__init__()
        self.number = number
        self.inverse = inverse
        self.remove_mode = remove_mode

    def forward(self, x):
        if self.inverse and self.remove_mode:
            x = x[:, :-self.number, :]
        elif self.inverse and not self.remove_mode:
            x = x[:, -self.number:, :]
        elif not self.inverse and self.remove_mode:
            x = x[:, self.number:, :]
        else:
            x = x[:, :self.number, :]
        return x


class TokenFnContext(nn.Module):
    def __init__(self, token_num=0, fn: nn.Module = nn.Identity(), token_fn: nn.Module = nn.Identity(), inverse=False):
        super(TokenFnContext, self).__init__()
        self.token_num = token_num
        self.fn = fn
        self.token_fn = token_fn
        self.inverse = inverse
        self.token_filter = TokenFilter(number=token_num, inverse=inverse, remove_mode=False)
        self.feature_filter = TokenFilter(number=token_num, inverse=inverse)

    def forward(self, x):
        tokens = self.token_filter(x)
        features = self.feature_filter(x)
        features = self.fn(features)
        if self.token_num == 0:
            return features

        tokens = self.token_fn(tokens)
        if self.inverse:
            x = torch.cat([features, tokens], dim=1)
        else:
            x = torch.cat([tokens, features], dim=1)
        return x


def init_weights(module):
    for n, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
