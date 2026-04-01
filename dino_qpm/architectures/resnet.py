import copy
import time

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision.models import get_model

# from scripts.modelExtensions.crossModelfunctions import init_experiment_stuff


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2',
           'wide_resnet50_3', 'wide_resnet50_4', 'wide_resnet50_5',
           'wide_resnet50_6', ]

from dino_qpm.architectures.FinalLayer import FinalLayer
from dino_qpm.architectures.utils import SequentialWithArgs

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, features=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x,  no_relu=False):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        if no_relu:
            return out
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, features=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        if features is None:
            self.conv3 = conv1x1(width, planes * self.expansion)
            self.bn3 = norm_layer(planes * self.expansion)
        else:
            self.conv3 = conv1x1(width, features)
            self.bn3 = norm_layer(features)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x,  no_relu=False, early_exit=False):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        if no_relu:
            return out
        return self.relu(out)


class ResNet(nn.Module, FinalLayer):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, changed_strides=False, model_type="qpm"):
        super(ResNet, self).__init__()

        self.model_type = model_type

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        widths = [64, 128, 256, 512]
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.sstride = 2
        if changed_strides:
            self.sstride = 1
        self.layer3 = self._make_layer(block, 256, layers[2], stride=self.sstride,
                                       dilate=replace_stride_with_dilation[1])
        self.stride = 2

        if changed_strides:
            self.stride = 1
        self.layer4 = self._make_layer(block, 512, layers[3], stride=self.stride,
                                       dilate=replace_stride_with_dilation[2])
        FinalLayer.__init__(self, num_classes, 512 *
                            block.expansion, arch="resnet50")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, last_block_f=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            krepeep = None
            if last_block_f is not None and _ == blocks - 1:
                krepeep = last_block_f
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, features=krepeep))

        return SequentialWithArgs(*layers)

    # Extended version of forward pass for EQPM
    def _forward(self, x, with_feature_maps=False, with_final_features=False, chunk_flips: bool = True):
        if chunk_flips:
            feature_maps = self.chunked_forward(x)

        else:
            feature_maps = self.non_chunked_forward(x)

        return self.transform_output(feature_maps, with_feature_maps,
                                     with_final_features)

    def get_flipped_feature_maps(self, x, flip_mode: str = "horizontal"):
        flipped_in = self.flip(x, flip_mode=flip_mode)
        flipped_feat_maps = self.get_feature_maps(flipped_in)
        reflipped_feature_maps = self.flip(
            flipped_feat_maps, flip_mode=flip_mode)

        return reflipped_feature_maps

    def get_double_flipped_feature_maps(self, x):
        double_flipped = self.flip(
            self.flip(x, flip_mode="horizontal"), flip_mode="vertical")
        double_flipped_feat_maps = self.get_feature_maps(double_flipped)
        reflipped_double_flipped_feat_maps = self.flip(self.flip(
            double_flipped_feat_maps, flip_mode="horizontal"), flip_mode="vertical")
        return reflipped_double_flipped_feat_maps

    def non_chunked_forward(self, x):
        device = x.device
        feature_maps = self.get_feature_maps(x).to("cpu")

        if self.model_type in ["eqpm-h", "eqpm-hv"]:
            feature_maps = torch.min(feature_maps, self.get_flipped_feature_maps(
                x, flip_mode="horizontal").to("cpu"))

        if self.model_type in ["eqpm-v", "eqpm-hv"]:
            feature_maps = torch.min(feature_maps, self.get_flipped_feature_maps(
                x, flip_mode="vertical").to("cpu"))

        if self.model_type == "eqpm-hv":
            feature_maps = torch.min(
                feature_maps, self.get_double_flipped_feature_maps(x).to("cpu"))

        return feature_maps.to(device)

    def chunked_forward(self, x):
        num_chunks = 1
        mul_feat_maps = [x]

        if self.model_type in ["eqpm-h", "eqpm-hv"]:
            mul_feat_maps.append(self.flip(x, flip_mode="horizontal"))
            num_chunks += 1

        if self.model_type in ["eqpm-v", "eqpm-hv"]:
            mul_feat_maps.append(self.flip(x, flip_mode="vertical"))
            num_chunks += 1

        # Append double flip only for eqpm-hv
        if self.model_type == "eqpm-hv":
            double_flipped = self.flip(
                self.flip(x, flip_mode="horizontal"), flip_mode="vertical")
            mul_feat_maps.append(double_flipped)
            num_chunks += 1

        if len(mul_feat_maps) == 1:
            input = mul_feat_maps[0]

        else:
            input = torch.cat(mul_feat_maps, dim=0)

        feature_maps = self.get_feature_maps(input)

        if num_chunks > 1:
            # Split feature maps into chunks
            feature_maps = feature_maps.view(num_chunks, -1,
                                             feature_maps.size(1),
                                             feature_maps.size(2),
                                             feature_maps.size(3))

            # Reflip
            feature_maps = self.reflip_feat_maps(feature_maps)

            # Combine feature maps by min operation
            feature_maps = torch.min(feature_maps, dim=0)[0]

        return feature_maps

    def reflip_feat_maps(self, feature_maps: torch.Tensor) -> torch.Tensor:
        # 1. Create a new list to hold the (new) re-flipped tensors
        reflipped_maps = []

        # 2. Add the original, unflipped map (index 0)
        #    We are appending the *original tensor*, not a new one.
        reflipped_maps.append(feature_maps[0])

        # 3. Handle horizontal flip (index 1 in 'h' and 'hv' modes)
        if self.model_type in ["eqpm-h", "eqpm-hv"]:
            # self.flip() creates a NEW tensor. Append it.
            reflipped_maps.append(
                self.flip(feature_maps[1], flip_mode="horizontal"))

        # 4. Handle vertical flip
        if self.model_type == "eqpm-v":
            # (index 1 in 'v' mode)
            reflipped_maps.append(
                self.flip(feature_maps[1], flip_mode="vertical"))

        elif self.model_type == "eqpm-hv":
            # (index 2 in 'hv' mode)
            # This assumes 'hv' mode always has 3 chunks [orig, h, v]
            reflipped_maps.append(
                self.flip(feature_maps[2], flip_mode="vertical"))

        # Handle double flip for 'hv' mode
        if self.model_type == "eqpm-hv":
            reflipped_maps.append(
                self.flip(self.flip(feature_maps[3], flip_mode="horizontal"), flip_mode="vertical"))

        # 5. Stack the list of tensors along dim 0 to create one
        #    NEW tensor. This is autograd-safe.
        return torch.stack(reflipped_maps, dim=0)

    def get_feature_maps(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feature_maps = self.layer4(x, no_relu=True)
        feature_maps = torch.functional.F.relu(feature_maps)

        return feature_maps

    @staticmethod
    def flip(x, flip_mode: str = "horizontal"):
        """
        Takes a batch of images or feature maps (batch_size, channels, height, width) as a tensor and then either flips it horizontally or vertically.
        """
        if flip_mode == "horizontal":
            x = torch.flip(x, [3])  # Flip along width dimension
        elif flip_mode == "vertical":
            x = torch.flip(x, [2])  # Flip along height dimension
        return x

    # Allow for accessing forward method in a inherited class
    forward = _forward


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        if kwargs["num_classes"] == 1000:
            state_dict["linear.weight"] = state_dict["fc.weight"]
            state_dict["linear.bias"] = state_dict["fc.bias"]
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_3(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-3 model
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 3
    return _resnet('wide_resnet50_3', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_4(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-4 model
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 4
    return _resnet('wide_resnet50_4', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_5(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-5 model
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 5
    return _resnet('wide_resnet50_5', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_6(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-6 model
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 6
    return _resnet('wide_resnet50_6', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
