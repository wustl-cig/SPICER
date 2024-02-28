import os
import pathlib
from argparse import ArgumentParser
import torch
from torch.nn import functional as F
import numpy as np


import matplotlib.pyplot as plt
# from basic_network import VarNet
from networks.network_n2n_tau_8x import SPNet
from tqdm import tqdm
import scipy.io as sio
from datetime import datetime
# from torch.utils.tensorboard import SummaryWriter
# from torch.utils.data import DataLoader
from dataset.pmri_fastmri_brain import RealMeasurement
from sigpy.mri.sim import birdcage_maps
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from utils.util import *
from utils.measures import *
from dataset.pmri_fastmri_brain import fmult, ftran
from utils.loss_functions import gradient_loss, forward_operator

if torch.cuda.is_available():
    device = 'cuda:2'
else:
    device = 'cpu'



def train(epoch):
    model.train()
    train_av_epoch_psnr_list = []
    train_av_epoch_loss_list = []
    train_av_epoch_ssim_list = []
    for iteration, samples in enumerate(iter_):
        x_hat, smps_hat, y, mask_m, mask_n = samples

        mask_m = mask_m.byte().to(device)
        mask_n = mask_n.byte().to(device)

        y = y.to(device)
        y_m = y * mask_m.to(device)
        y_n = y * mask_n.to(device)

        ny = y_m.shape[-2]
        acs_percentage = 0.2
        ACS_size = ((ny // 2) - (int(ny * acs_percentage * (2 / acceleration_factor)) // 2)) * 2

        output_m, smap_m = model(torch.view_as_real(y_m), mask_m, ACS_center=(ny // 2), ACS_size=ACS_size)
        output_n, smap_n = model(torch.view_as_real(y_n), mask_n, ACS_center=(ny // 2), ACS_size=ACS_size)

        smap_m_1 = torch.view_as_complex(smap_m.squeeze())
        smap_n_1 = torch.view_as_complex(smap_n.squeeze())

        h_output_n = fmult(torch.view_as_complex(output_m), smap_m_1, mask_n)
        h_output_m = fmult(torch.view_as_complex(output_n), smap_n_1, mask_m)
        smap_m_loss = smap_m.squeeze().permute(0, 3, 1, 2)


        loss = ((F.mse_loss(torch.view_as_real(h_output_m).float().squeeze(), torch.view_as_real(y_m).float().squeeze()).float() + F.mse_loss(torch.view_as_real(h_output_n).float().squeeze(),
                                                                                  torch.view_as_real(y_n).float().squeeze()).float())/2) + 0.001 * gradient_loss(
            smap_m_loss)
        loss = loss.float()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output_show = output_m.to('cpu').detach()
        output_show = complex_abs(output_show).squeeze()
        output_show = normlize(output_show)
        target_show = torch.abs(x_hat.squeeze())
        target_show = normlize(target_show)

        train_av_epoch_psnr_list.append(compare_psnr(output_show, target_show))
        train_av_epoch_ssim_list.append(compare_ssim(output_show.unsqueeze(0).unsqueeze(0), target_show.unsqueeze(0).unsqueeze(0)))
        train_av_epoch_loss_list.append(loss.item())
    psnr_value = np.mean(train_av_epoch_psnr_list)
    ssim_value = np.mean(train_av_epoch_ssim_list)
    loss_value = np.mean(train_av_epoch_loss_list)
    print('The PSNR value for N2N output is {}'.format(psnr_value))
    print('The SSIM value for N2N output is {}'.format(ssim_value))
    print('training loss at epoch {} is {}'.format(epoch, loss_value))

def val(epoch):
    model.eval()
    eval_av_epoch_psnr_list = []
    eval_av_epoch_loss_list = []
    eval_av_epoch_ssim_list = []
    for iteration, samples in enumerate(valloader):
        dicom, x0, y_input, smps_input, mask_input = samples

        mask_m = mask_input.byte().to(device)


        y_m = y_input.to(device)
        x_hat = dicom

        ny = y_m.shape[-2]
        acs_percentage = 0.2
        ACS_size = ((ny // 2) - (int(ny * acs_percentage * (2 / acceleration_factor)) // 2)) * 2

        output_m, smap_m = model(torch.view_as_real(y_m), mask_m, ACS_center=(ny // 2), ACS_size=ACS_size)

        output_show = output_m.to('cpu').detach()
        output_show = complex_abs(output_show).squeeze()
        output_show = normlize(output_show)
        target_show = torch.abs(x_hat.squeeze())
        target_show = normlize(target_show)

        eval_av_epoch_psnr_list.append(compare_psnr(output_show, target_show))
        eval_av_epoch_ssim_list.append(
            compare_ssim(output_show.unsqueeze(0).unsqueeze(0), target_show.unsqueeze(0).unsqueeze(0)))

    print('val:The PSNR value for N2N output is {}'.format(np.mean(eval_av_epoch_psnr_list)))
    print('val:The SSIM value for N2N output is {}'.format(np.mean(eval_av_epoch_ssim_list)))
    snr_best.append(np.mean(eval_av_epoch_ssim_list))

    # save model every 50 epoches
    if epoch % 5 == 0 and epoch > 0:
        print('save the model at epoch {}'.format(epoch))
        model_dir = './model/{}'.format(model_name)
        if not (os.path.exists(model_dir)): os.makedirs(model_dir)
        torch.save(model.state_dict(), "{0}/N2N_{1:03d}.pth".format(save_root, epoch))
    elif np.mean(eval_av_epoch_psnr_list) >= np.max(np.array(snr_best)):
        print('save the model at epoch {}'.format(epoch))
        model_dir = './model/{}'.format(model_name)
        if not (os.path.exists(model_dir)): os.makedirs(model_dir)
        torch.save(model.state_dict(), "{0}/N2N_{1:03d}.pth".format(save_root, epoch))




if __name__ == "__main__":
    # ------------
    # parameters
    # ------------

    now = datetime.now()
    model_name = 'SPICER_fastmri'
    batch = 1
    workers = 2
    epoch_number = 200
    data_len = 1
    acceleration_factor = 8
    save_root = './model_zoo/SPICER_fastmri_%x/%s' % (acceleration_factor, str(now.strftime("%d-%b-%Y-%H-%M-%S")))
    if not (os.path.exists(save_root)): os.makedirs(save_root)


    dataset = RealMeasurement(
        idx_list=range(564, 1355),
        acceleration_rate=acceleration_factor,
        is_return_y_smps_hat=True,
        mask_pattern='uniformly_cartesian',
        smps_hat_method='eps',
    )
    trainloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    val_dataset = RealMeasurement(
        idx_list=range(1355,1377),
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

    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0)
    snr_best = []

    for epoch in range(epoch_number):
        iter_ = tqdm(trainloader, desc='Train [%.3d/%.3d]' % (epoch, epoch_number), total=len(dataset))

        print('Epoch {}'.format(epoch))
        train(epoch)
        with torch.no_grad():
            val(epoch)

        lr_scheduler(optimizer, epoch)
