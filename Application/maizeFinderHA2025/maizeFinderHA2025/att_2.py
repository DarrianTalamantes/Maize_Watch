import torch.hub
from torchvision import transforms, datasets, models
from torch import nn
# Add the custom model classes
class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size

        assert kernel_size % 2 == 1, "Odd kernel size required"
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=int((kernel_size-1)/2))

    def forward(self, x):
        max_pool = self.agg_channel(x, "max")
        avg_pool = self.agg_channel(x, "avg")
        pool = torch.cat([max_pool, avg_pool], dim=1)
        conv = self.conv(pool)
        conv = conv.repeat(1, x.size()[1], 1, 1)
        att = torch.sigmoid(conv)
        return att

    def agg_channel(self, x, pool="max"):
        b, c, h, w = x.size()
        x = x.view(b, c, h*w)
        x = x.permute(0, 2, 1)
        if pool == "max":
            x = nn.functional.max_pool1d(x, c)
        elif pool == "avg":
            x = nn.functional.avg_pool1d(x, c)
        x = x.permute(0, 2, 1)
        x = x.view(b, 1, h, w)
        return x

class ECA_SA(nn.Module):
    def __init__(self, n_channels_in, k_size, kernel_size):
        super(ECA_SA, self).__init__()
        self.n_channels_in = n_channels_in
        self.k_size = k_size
        self.kernel_size = kernel_size

        self.eca_layer = eca_layer(n_channels_in, k_size)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, f):
        chan_att = self.eca_layer(f)
        spat_att = self.spatial_attention(chan_att)
        fpp = spat_att * chan_att
        return fpp

class ECA_SA_ResNeXtModel(nn.Module):
    def __init__(self, num_classes=4):
        super(ECA_SA_ResNeXtModel, self).__init__()
        # Use torchvision's resnext instead of loading from torch.hub to avoid errors
        resnext = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
        self.base = nn.Sequential(*list(resnext.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.ECA_SA = ECA_SA(2048, k_size=3, kernel_size=3)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Sequential(
            nn.Linear(2048, num_classes)
        )

    def forward(self, input):
        features = self.base(input)
        ECA_SA_features = self.ECA_SA(features)
        x = features + ECA_SA_features
        x = self.avgpool(x)
        x = x.reshape(-1, 2048)  # flattening the tensor
        out = self.fc(x)
        return out