from torch.utils.data import Dataset
import torch

import numpy as np
import scipy.io as scio


class TrainingItem:
    def __init__(self, well_data, vp_init, vs_init, rho_init, d):
        self.well_data = well_data[:, 1:]
        self.init_data = np.array([vp_init, vs_init, rho_init]).transpose()
        self.d = np.vstack([d, d[-1, :]])   # d
        self.masked_index = None
        self.masked_d = None

    def gen_mask(self, seq_len, mask_prob=0.15):
        n_mask = int((seq_len - 1) * mask_prob)
        self.masked_index = np.random.randint(0, seq_len - 1, n_mask)
        self.masked_d = self.d.copy()
        self.masked_d[self.masked_index, :] = 0

    def to_torch(self):
        output = {"well_data": self.well_data,
                  "init_data": self.init_data,
                  "d": self.d,
                  "masked_index": self.masked_index,
                  "masked_d": self.masked_d}

        return {key: torch.tensor(value) for key, value in output.items()}


class TrainingData:
    """
    训练数据，从mat文件读取
    """
    def __init__(self, training_data_path):
        data = scio.loadmat(training_data_path)
        self.data = data
        self.well_data = data['wellData'].astype(np.float32)
        self.d = data['prestackFreeNoise']
        self.d_noise = data['prestackNoise']

        self.vp_init = data['vp_init'].astype(np.float32)
        self.vs_init = data['vs_init'].astype(np.float32)
        self.rho_init = data['rho_init'].astype(np.float32)
        self.depth = data['depth'].astype(np.float32)

        self.wavelet = data['wavelet'].astype(np.float32)
        self.dt = data['dt']

    def get_data(self, index) -> TrainingItem:
        well_data = self.well_data[:, index, :]
        vp_init = self.vp_init[:, index]
        vs_init = self.vs_init[:, index]
        rho_init = self.rho_init[:, index]
        d = self.d[0, index].astype(np.float32)
        item = TrainingItem(well_data, vp_init, vs_init, rho_init, d)
        return item

    @property
    def seq_len(self):
        return self.well_data.shape[0]

    @property
    def angle_num(self):
        return self.d[0, 0].shape[1]

    @property
    def length(self):
        return self.well_data.shape[1]


class BERTDataset(Dataset):
    def __init__(self, training_data_path, mask_prob=0.15):
        self.training_data = TrainingData(training_data_path)
        self.seq_len = self.training_data.seq_len
        self.angle_num = self.training_data.angle_num
        self.mask_prob = mask_prob

    def __len__(self):
        return self.training_data.length

    def __getitem__(self, index):
        item = self.training_data.get_data(index)
        item.gen_mask(self.seq_len, mask_prob=self.mask_prob)

        return item.to_torch()
