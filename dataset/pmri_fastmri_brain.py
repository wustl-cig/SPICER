import h5py
import os
import numpy as np
import tqdm
import torch
import tifffile
import pandas
from torch.utils.data import Dataset

#### ROOT_PATH is where the fastmri dataset stored ####
ROOT_PATH = './dataset/fastmri_brain_multicoil/'

#### DATASHEET_PATH is where the CSV datasheet stored ####
DATASHEET_PATH = './dataset/'
DATASHEET = pandas.read_csv(os.path.join(DATASHEET_PATH, 'fastmri_brain_multicoil.csv'))


def INDEX2_helper(idx, key_):
    file_id_df = DATASHEET[key_][DATASHEET['INDEX'] == idx]

    assert len(file_id_df.index) == 1

    return file_id_df[idx]


INDEX2FILE = lambda idx: INDEX2_helper(idx, 'FILE')

def INDEX2DROP(idx):
    ret = INDEX2_helper(idx, 'DROP')

    if ret in ['0', 'false', 'False', 0.0]:
        return False
    else:
        return True

def INDEX2SLICE_START(idx):
    ret = INDEX2_helper(idx, 'SLICE_START')

    if isinstance(ret, np.float64) and ret >= 0:
        return int(ret)
    else:
        return None


def INDEX2SLICE_END(idx):
    ret = INDEX2_helper(idx, 'SLICE_END')

    if isinstance(ret, np.float64) and ret >= 0:
        return int(ret)
    else:
        return None


def ftran(y, smps, mask):
    """
    compute adjoint of fast MRI, x = smps^H F^H mask^H x

    :param y: under-sampled measurements, shape: batch, coils, width, height; dtype: complex
    :param smps: sensitivity maps, shape: batch, coils, width, height; dtype: complex
    :param mask: sampling mask, shape: batch, width, height; dtype: float/bool
    :return: zero-filled image
    """

    # mask^H
    y = y * mask.unsqueeze(1)

    # F^H
    y = torch.fft.ifftshift(y, [-2, -1])
    x = torch.fft.ifft2(y, norm='ortho')
    x = torch.fft.fftshift(x, [-2, -1])

    # smps^H
    x = x * torch.conj(smps)
    x = x.sum(1)

    return x


def fmult(x, smps, mask):
    """
    compute forward of fast MRI, y = mask F smps x

    :param x: groundtruth or estimated image, shape: batch, width, height; dtype: complex
    :param smps: sensitivity maps, shape: batch, coils, width, height; dtype: complex
    :param mask: sampling mask, shape: batch, width, height; dtype: float/bool
    :return: undersampled measurement
    """

    # smps
    x = x.unsqueeze(1)
    y = x * smps

    # F
    y = torch.fft.ifftshift(y, [-2, -1])
    y = torch.fft.fft2(y, norm='ortho')
    y = torch.fft.fftshift(y, [-2, -1])

    # mask
    mask = mask.unsqueeze(1)
    y = y * mask

    return y


def uniformly_cartesian_mask(img_size, acceleration_rate, acs_percentage: float = 0.2, randomly_return: bool = False):

    ny = img_size[-1]

    ACS_START_INDEX = (ny // 2) - (int(ny * acs_percentage * (2 / acceleration_rate)) // 2)
    ACS_END_INDEX = (ny // 2) + (int(ny * acs_percentage * (2 / acceleration_rate)) // 2)

    if ny % 2 == 0:
        ACS_END_INDEX -= 1

    mask = np.zeros(shape=(acceleration_rate,) + img_size, dtype=np.float32)
    mask[..., ACS_START_INDEX: (ACS_END_INDEX + 1)] = 1

    for i in range(ny):
        for j in range(acceleration_rate):
            if i % acceleration_rate == j:
                mask[j, ..., i] = 1

    if randomly_return:
        mask = mask[np.random.randint(0, acceleration_rate)]
    # else:
    #     mask = mask[0]

    return mask[0], mask[acceleration_rate// 2]


_mask_fn = {
    'uniformly_cartesian': uniformly_cartesian_mask
}


def addwgn(x: torch.Tensor, input_snr):
    noiseNorm = torch.norm(x.flatten()) * 10 ** (-input_snr / 20)

    noise = torch.randn(x.size()).to(x.device)

    noise = noise / torch.norm(noise.flatten()) * noiseNorm

    y = x + noise
    return y


def check_and_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def np_normalize_to_uint8(x):
    x -= np.amin(x)
    x /= np.amax(x)

    x = x * 255
    x = x.astype(np.uint8)

    return x


def load_real_dataset_handle(
        idx,
        acceleration_rate: int = 1,
        is_return_y_smps_hat: bool = False,
        mask_pattern: str = 'uniformly_cartesian',
        smps_hat_method: str = 'eps',
):

    root_path = os.path.join(ROOT_PATH, 'real')
    check_and_mkdir(root_path)

    y_h5 = os.path.join(ROOT_PATH, INDEX2FILE(idx) + '.h5')

    meas_path = os.path.join(root_path, "acceleration_rate_%d_smps_hat_method_%s" % (
        acceleration_rate, smps_hat_method))
    check_and_mkdir(meas_path)

    x_hat_path = os.path.join(meas_path, 'x_hat')
    check_and_mkdir(x_hat_path)

    x_hat_h5 = os.path.join(x_hat_path, INDEX2FILE(idx) + '.h5')

    smps_hat_path = os.path.join(meas_path, 'smps_hat')
    check_and_mkdir(smps_hat_path)

    smps_hat_h5 = os.path.join(smps_hat_path, INDEX2FILE(idx) + '.h5')

    mask_path = os.path.join(meas_path, 'mask')
    check_and_mkdir(mask_path)

    mask_h5 = os.path.join(mask_path, INDEX2FILE(idx) + '.h5')

    if not os.path.exists(x_hat_h5):

        with h5py.File(y_h5, 'r') as f:
            y = f['kspace'][:]

            # Normalize the kspace to 0-1 region
            for i in range(y.shape[0]):
                y[i] /= np.amax(np.abs(y[i]))

        if not os.path.exists(mask_h5):

            _, _, n_x, n_y = y.shape
            if acceleration_rate > 1:
                mask = _mask_fn[mask_pattern]((n_x, n_y), acceleration_rate)

            else:
                mask = np.ones(shape=(n_x, n_y), dtype=np.float32)

            mask = np.expand_dims(mask, 0)  # add batch dimension
            mask = torch.from_numpy(mask)

            with h5py.File(mask_h5, 'w') as f:
                f.create_dataset(name='mask', data=mask)

        else:

            with h5py.File(mask_h5, 'r') as f:
                mask = f['mask'][:]

        if not os.path.exists(smps_hat_h5):

            os.environ['CUPY_CACHE_DIR'] = '/tmp/cupy'
            os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba'
            from sigpy.mri.app import EspiritCalib
            from sigpy import Device
            import cupy

            num_slice = y.shape[0]
            iter_ = tqdm.tqdm(range(num_slice), desc='[%d, %s] Generating coil sensitivity map (smps_hat)' % (
                idx, INDEX2FILE(idx)))

            smps_hat = np.zeros_like(y)
            for i in iter_:
                tmp = EspiritCalib(y[i] * mask, device=Device(0), show_pbar=False).run()
                tmp = cupy.asnumpy(tmp)
                smps_hat[i] = tmp

            with h5py.File(smps_hat_h5, 'w') as f:
                f.create_dataset(name='smps_hat', data=smps_hat)

            tmp = np.ones(shape=smps_hat.shape, dtype=np.uint8)
            for i in range(tmp.shape[0]):
                for j in range(tmp.shape[1]):
                    tmp[i, j] = np_normalize_to_uint8(abs(smps_hat[i, j]))
            tifffile.imwrite(smps_hat_h5.replace('.h5', '_qc.tiff'), data=tmp, compression='zlib', imagej=True)

        else:

            with h5py.File(smps_hat_h5, 'r') as f:
                smps_hat = f['smps_hat'][:]

        y = torch.from_numpy(y)
        smps_hat = torch.from_numpy(smps_hat)

        x_hat = ftran(y, smps_hat, mask)

        with h5py.File(x_hat_h5, 'w') as f:
            f.create_dataset(name='x_hat', data=x_hat)

        tmp = np.ones(shape=x_hat.shape, dtype=np.uint8)
        for i in range(x_hat.shape[0]):
            tmp[i] = np_normalize_to_uint8(abs(x_hat[i]).numpy())
        tifffile.imwrite(x_hat_h5.replace('.h5', '_qc.tiff'), data=tmp, compression ='zlib', imagej=True)

    ret = {
        'x_hat': x_hat_h5
    }
    if is_return_y_smps_hat:
        ret.update({
            'smps_hat': smps_hat_h5,
            'y': y_h5,
            'mask': mask_h5
        })

    return ret



class RealMeasurement(Dataset):
    def __init__(
            self,
            idx_list,
            acceleration_rate,
            is_return_y_smps_hat: bool = False,
            mask_pattern: str = 'uniformly_cartesian',
            smps_hat_method: str = 'eps',
    ):

        self.__index_maps = []
        for idx in idx_list:
            if INDEX2DROP(idx):
                continue

            ret = load_real_dataset_handle(
                idx,
                1,
                is_return_y_smps_hat,
                mask_pattern,
                smps_hat_method
            )

            with h5py.File(ret['x_hat'], 'r') as f:
                num_slice = f['x_hat'].shape[0]

            if INDEX2SLICE_START(idx) is not None:
                slice_start = INDEX2SLICE_START(idx)
            else:
                slice_start = 0

            if INDEX2SLICE_END(idx) is not None:
                slice_end = INDEX2SLICE_END(idx)
            else:
                slice_end = num_slice - 5

            for s in range(slice_start, slice_end):
                self.__index_maps.append([ret, s])

            self.acceleration_rate = acceleration_rate

        self.is_return_y_smps_hat = is_return_y_smps_hat

    def __len__(self):
        return len(self.__index_maps)

    def __getitem__(self, item):

        ret, s = self.__index_maps[item]

        with h5py.File(ret['x_hat'], 'r', swmr=True) as f:
            x_hat = f['x_hat'][s]

        if self.is_return_y_smps_hat:
            with h5py.File(ret['smps_hat'], 'r', swmr=True) as f:
                smps_hat = f['smps_hat'][s]

            with h5py.File(ret['y'], 'r', swmr=True) as f:
                y = f['kspace'][s]

                # Normalize the kspace to 0-1 region
                y /= np.amax(np.abs(y))

            # with h5py.File(ret['mask'], 'r', swmr=True) as f:
            #     mask = f['mask'][0]
            nx = y.shape[-2]
            ny = y.shape[-1]
            mask1, mask2 = _mask_fn['uniformly_cartesian']((nx, ny), self.acceleration_rate)

            return x_hat, smps_hat, y, mask1, mask2

        else:

            return x_hat,
