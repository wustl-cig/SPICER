import os
import pathlib
from argparse import ArgumentParser
import torch
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt

from networks.SPICER_fastmri_network import SPNet
from tqdm import tqdm
import scipy.io as sio
from datetime import datetime
from dataset.pmri_fastmri_brain import RealMeasurement
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from utils.util import *
from utils.measures import *
from dataset.pmri_fastmri_brain import fmult, ftran

if torch.cuda.is_available():
    device = 'cuda:1'
else:
    device = 'cpu'

def val():
    model.eval()
    eval_av_epoch_psnr_list = []
    eval_av_epoch_ssim_list = []
    eval_av_epoch_nmse_list = []
    for iteration, samples in enumerate(valloader):
        x_hat, smps_hat, y, mask_m, mask_n = samples
        mask_m = mask_m.byte().to(device)

        y = y.to(device)
        y_m = y * mask_m.to(device)

        ny = y_m.shape[-2]
        acs_percentage = 0.2
        ACS_size = ((ny // 2) - (int(ny * acs_percentage * (2 / acceleration_factor)) // 2)) * 2

        output_m, smap_m = model(torch.view_as_real(y_m), mask_m, ACS_center=(ny // 2), ACS_size=ACS_size)

        ###
        brain_mask = torch.view_as_real(smps_hat[0]).to(device)[0]
        brain_mask[brain_mask != 0] = 1

        output_m = complex_mul(output_m, complex_conj(brain_mask)).sum(
            dim=0, keepdim=True
        )
        output_show = output_m.to('cpu').detach()

        output_show = complex_abs(output_show)[0]
        target_show = torch.abs(x_hat.squeeze())

        output_show = normlize(center_crop(output_show, [400, 320]))
        target_show = normlize(center_crop(target_show, [400, 320]))

        eval_av_epoch_nmse_list.append(compare_nmse(output_show, target_show))
        eval_av_epoch_psnr_list.append(compare_psnr(output_show, target_show))
        eval_av_epoch_ssim_list.append(compare_ssim(output_show.unsqueeze(0), target_show.unsqueeze(0).unsqueeze(0)))

    print('val:The PSNR value for N2N output is {}'.format(np.mean(eval_av_epoch_psnr_list)))
    print('val:The SSIM value for N2N output is {}'.format(np.mean(eval_av_epoch_ssim_list)))
    print('val:The NMSE value for N2N output is {}'.format(np.mean(eval_av_epoch_nmse_list)))






if __name__ == "__main__":
    # ------------
    # parameters
    # ------------

    now = datetime.now()
    batch = 1
    workers = 2
    data_len = 1
    acceleration_factor = 8

    model_path = './model_zoo/SPICER_fastmri_%dx/model.pth' %(acceleration_factor)




    val_dataset = RealMeasurement(
        idx_list=range(1355, 1377),
        acceleration_rate=acceleration_factor,
        is_return_y_smps_hat=True,
        mask_pattern='uniformly_cartesian',
        smps_hat_method='eps',
    )
    valloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = SPNet(
        num_cascades=6,
        pools=4,
        chans=18,
        sens_pools=4,
        sens_chans=8,
    ).to(device)
    print(device)
    with torch.no_grad():
        test_state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(test_state_dict)
        val()