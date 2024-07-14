from ._base import BaseDistiller
from .registry import register_distiller
from .operators import SepConv, PatchMerging, GAP1d, TokenFnContext, TokenFilter, init_weights
from .distillation_losses import ofa_loss

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from timm.models.vision_transformer import Block


@register_distiller
class OFAKD(BaseDistiller):
    requires_feature = True

    def __init__(self, student, teacher, criterion, args):
        super(OFAKD, self).__init__(student, teacher, criterion, args)

        self.projector = nn.ModuleDict()

        is_cnn_student = self.student.is_cnn_model()
        _, feature_dim_t = self.teacher.stage_info(-1)
        _, feature_dim_s = self.student.stage_info(-1)

        for stage in self.args.ofa_stages:
            _, size_s = self.student.stage_info(stage)
            if is_cnn_student:
                in_chans, _, _ = size_s
                if stage != 4:
                    down_sample_blk_num = 4 - stage
                    down_sample_blks = []
                    for i in range(down_sample_blk_num):
                        if i == down_sample_blk_num - 1:
                            out_chans = max(feature_dim_s, feature_dim_t)
                        else:
                            out_chans = in_chans * 2
                        down_sample_blks.append(SepConv(in_chans, out_chans))
                        in_chans *= 2
                else:
                    down_sample_blks = [nn.Conv2d(in_chans, max(feature_dim_s, feature_dim_t), 1, 1, 0)]

                projector = nn.Sequential(
                    *down_sample_blks,
                    nn.AdaptiveMaxPool2d(1),
                    nn.Flatten(),
                    nn.Linear(max(feature_dim_s, feature_dim_t), self.args.num_classes)
                )
            else:
                patch_num, embed_dim = size_s
                cls_token_num = getattr(self.student, 'num_tokens', 1)
                final_patch_grid = 7
                patch_grid = int(patch_num ** .5)
                merge_num = max(int(np.log2(patch_grid / final_patch_grid)), 0)
                merger_modules = []
                for i in range(merge_num):
                    if i == 0:  # proj to feature_dim_s
                        merger_modules.append(
                            PatchMerging(input_resolution=(patch_grid // 2 ** i, patch_grid // 2 ** i),
                                         dim=embed_dim,
                                         out_dim=feature_dim_s,
                                         act_layer=nn.GELU))
                    else:
                        merger_modules.append(
                            PatchMerging(input_resolution=(patch_grid // 2 ** i, patch_grid // 2 ** i),
                                         dim=feature_dim_s,
                                         out_dim=feature_dim_s,
                                         act_layer=nn.GELU if i != merge_num - 1 else nn.Identity))
                patch_merger = nn.Sequential(*merger_modules)
                blocks = nn.Sequential(
                    *[Block(dim=feature_dim_s, num_heads=4) for _ in range(max(4 - stage, 1))]  # todo: check this
                )
                if cls_token_num != 0:
                    get_feature = nn.Sequential(
                        TokenFilter(cls_token_num, remove_mode=False),  # todo: token_num > 1
                        nn.Flatten()
                    )
                else:
                    get_feature = GAP1d()
                projector = nn.Sequential(
                    TokenFnContext(cls_token_num, patch_merger),
                    blocks,
                    get_feature,
                    nn.Linear(feature_dim_s, self.args.num_classes)
                )
            self.projector[str(stage)] = projector
            self.projector.apply(init_weights)

    def forward(self, image, label):
        with torch.no_grad():
            self.teacher.eval()
            logits_teacher = self.teacher(image)

        logits_student, features_student = self.student(image, requires_feature=self.requires_feature)

        num_classes = logits_student.size(-1)
        target_mask = F.one_hot(label, num_classes)

        ofa_losses = []
        for stage in self.args.ofa_stages:
            idx_s, _ = self.student.stage_info(stage)
            feat_s = features_student[idx_s]
            logits_student_head = self.projector[str(stage)](feat_s)

            ofa_losses.append(
                ofa_loss(logits_student_head, logits_teacher, target_mask, self.args.ofa_eps,
                         self.args.ofa_temperature))

        loss_ofa = self.args.ofa_loss_weight * sum(ofa_losses)

        loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)
        loss_kd = self.args.kd_loss_weight * ofa_loss(logits_student, logits_teacher, target_mask,
                                                      self.args.ofa_eps, self.args.ofa_temperature)
        losses_dict = {
            "loss_gt": loss_gt,
            "loss_kd": loss_kd,
            "loss_ofa": loss_ofa
        }
        return logits_student, losses_dict
