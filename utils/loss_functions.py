import torch
from utils.util import *


def gradient_loss(s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
    dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])

    if penalty == 'l2':
        dy = dy * dy
        dx = dx * dx

    d = torch.mean(dx) + torch.mean(dy)
    return d / 2.0


def forward_operator(out_img, smap, mask):
    out_img_multi = complex_mul(out_img, smap)
    out_kspace_multi = fft2c(out_img_multi)
    if mask.size(dim=-1) != 2:
        mask = torch.stack([mask, mask], dim=-1)
    h_output = out_kspace_multi * mask + 0.0
    return h_output


def grad_g_operator(out_img, smap, mask):
    out_img_multi = complex_mul(out_img, smap)
    grad_g_output = fft2c(out_img_multi)
    return grad_g_output

def transpose_operator(out_kspace, smap, mask):
    out_img_multi = ifft2c(out_kspace)
    h_trans_output = complex_mul(out_img_multi, complex_conj(smap)).sum(
        dim=1, keepdim=True
    )
    return h_trans_output
