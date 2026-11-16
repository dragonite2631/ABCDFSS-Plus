import torch.nn as nn
import torch.nn.functional as F


class SpatialContextAdapter(nn.Module):
    """
    Lightweight Residual Spatial-Context Adapter (Improvement 2):
    Conv1x1 -> BN -> ReLU -> Residual DWConv3x3 -> Conv1x1
    Adds only 9*C + 2*C parameters per layer (e.g. 704 params for C=64)
    to capture local spatial neighborhood without quadratic parameter explosion.
    """

    def __init__(self, in_channels, config_model):
        super().__init__()
        out_channels = config_model.out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dwconv = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False
        )
        self.dw_bn = nn.BatchNorm2d(out_channels)
        self.linear = nn.Conv2d(out_channels, out_channels, 1)
        self.prepend_relu = config_model.prepend_relu
        self.append_normalize = config_model.append_normalize

    def forward(self, x):
        if self.prepend_relu:
            x = F.relu(x)
        h = F.relu(self.bn(self.conv(x)))
        # Residual depthwise spatial mixing
        h = h + F.relu(self.dw_bn(self.dwconv(h)))
        out = self.linear(h)
        if self.append_normalize:
            out = F.normalize(out, p=2, dim=1)
        return out
