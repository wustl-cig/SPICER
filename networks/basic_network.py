"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""


"""
Varnet for noise2noise training without sensitivity map estimation module
Yuyang Hu @ wustl
"""

import math
from typing import List, Tuple
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet import Unet
from utils.util import *

class NormUnet(nn.Module):
    """
    Normalized U-Net model.
    This is the same as a regular U-Net, but with normalization applied to the
    input before the U-Net. This keeps the values more numerically stable
    during training.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()


        self.unet = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)

        mean = x.mean(dim=2).view(b, c, 1, 1)
        std = x.std(dim=2).view(b, c, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / (std), mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")

        # get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)

        x = self.unet(x)

        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)


        return x


class VarNet(nn.Module):
    """
    A full variational network model.
    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBock.
    """

    def __init__(
        self,
        chans: int = 18,
        pools: int = 4,

    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
        """
        super().__init__()
        self.model = NormUnet(chans, pools)
    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        x = ifft2c(x)
        return complex_mul(x, complex_conj(sens_maps)).sum(
            dim=1, keepdim=True
        )
    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        return x / rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fft2c(complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        x = ifft2c(x)
        return complex_mul(x, complex_conj(sens_maps)).sum(
            dim=1, keepdim=True
        )

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor, espirit_map: torch.Tensor) -> torch.Tensor:
        sens_maps = espirit_map
        kspace_pred = self.sens_expand(self.model(self.sens_reduce(masked_kspace, sens_maps)), sens_maps)
        single_image = complex_abs(ifft2c(kspace_pred))
        smap_complex = torch.view_as_complex(sens_maps)
        total_image = torch.sum(single_image * smap_complex.conj(), dim=1, keepdim=False)

        return torch.absolute(total_image), sens_maps