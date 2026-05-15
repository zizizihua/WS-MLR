import torch
from torch import nn
import torch.nn.functional as F

from .backbone.resnet import resnet101


class ImageClassifier(nn.Module):
    def __init__(self, classNum, pretrained=True, freezeBackbone=False):
        super().__init__()

        self.classNum = classNum
        self.backbone = resnet101(pretrained=pretrained, avg_pool=False)

        self.onebyone_conv = nn.Conv2d(2048, self.classNum, 1)
        alpha = torch.ones(1, self.classNum, dtype=torch.float)
        self.register_buffer('alpha', alpha)

        if freezeBackbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x, fw_cam=False):
        feats = self.backbone(x)
        CAM = self.onebyone_conv(feats)
        CAM = torch.where(CAM > 0, CAM * self.alpha.view(1, -1, 1, 1), CAM)
        logits = F.adaptive_avg_pool2d(CAM, 1).squeeze(-1).squeeze(-1)

        if fw_cam:
            return logits, CAM

        return logits
