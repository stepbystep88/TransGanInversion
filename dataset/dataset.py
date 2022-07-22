from torch.utils.data import Dataset
import torch

import numpy as np
import scipy.io as scio
from sklearn.preprocessing import MinMaxScaler


class TrainingItem:
    def __init__(self, well_data, init_data, d, d_noise):
        self.well_data = well_data
        self.init_data = init_data
        self.d = np.vstack([d, d[-1, :]])  # d
        self.d_noise = np.vstack([d_noise, d_noise[-1, :]])
        # self.masked_index = None
        # self.masked_d = None
        # self.mask = None

    # def gen_mask(self, seq_len, mask_prob=0.2, mask_noise_prob=0.5):
    #     n_mask = int((seq_len - 1) * mask_prob)
    #     self.masked_index = np.random.randint(0, seq_len - 1, n_mask)
    #
    #     self.mask = np.ones((seq_len, seq_len))
    #     self.mask[:, self.masked_index] = 0
    #     self.mask[self.masked_index, :] = 0
    #
    #     self.masked_d = self.d.copy()
    #     self.masked_d[self.masked_index, :] = 0
    #     n_mask_noise = int(n_mask * mask_noise_prob)
    #     noise_index = np.random.choice(self.masked_index, n_mask_noise, replace=False)
    #     self.masked_d[noise_index, :] = self.d_noise[noise_index, :]

    def to_torch(self):
        output = {"well_data": self.well_data,
                  "init_data": self.init_data,
                  "d": self.d,
                  "d_noise": self.d_noise
                  }
        # "mask": self.mask,
        # "masked_index": self.masked_index,
        # "masked_d": self.masked_d

        return {key: torch.tensor(value) for key, value in output.items()}


class TrainingData:
    """
    训练数据，从mat文件读取
    """
    def __init__(self, training_data_path, need_normalize=True, well_trans=None):
        data = scio.loadmat(training_data_path)
        self.data = data
        self.well_data = data['wellData'][:, 1:, 1:].astype(np.float32)

        vp_init = data['vp_init'][:, 1:].astype(np.float32)
        vs_init = data['vs_init'][:, 1:].astype(np.float32)
        rho_init = data['rho_init'][:, 1:].astype(np.float32)
        self.init_data = np.transpose(np.array([vp_init, vs_init, rho_init]), (1, 2, 0))

        self.d = data['prestackFreeNoise'][:, 1:]
        self.d_noise = data['prestackNoise'][:, 1:]

        self.depth = data['depth'][:, 1:].astype(np.float32)
        self.wavelet = data['wavelet'].astype(np.float32)
        self.dt = data['dt']

        self.well_trans = well_trans
        if need_normalize:
            self.normalize()

    def normalize(self):
        self.well_trans = MinMaxScaler(feature_range=(-1, 1))
        seq_len = self.well_data.shape[0]
        self.well_data = np.reshape(self.well_trans.fit_transform(np.reshape(self.well_data, (-1, 3))), (seq_len, -1, 3))
        self.init_data = np.reshape(self.well_trans.transform(np.reshape(self.init_data, (-1, 3))), (seq_len, -1, 3))

    def get_data(self, index) -> TrainingItem:
        well_data = self.well_data[:, index, :]
        init_data = self.init_data[:, index, :]
        d = self.d[0, index].astype(np.float32)
        d_noise = self.d[0, index].astype(np.float32)
        item = TrainingItem(well_data, init_data, d, d_noise)

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
    def __init__(self, training_data_path, mask_prob=0.2, mask_noise_prob=0.5):
        self.training_data = TrainingData(training_data_path)
        self.seq_len = self.training_data.seq_len
        self.angle_num = self.training_data.angle_num
        self.mask_prob = mask_prob
        self.mask_noise_prob = mask_noise_prob

    def __len__(self):
        return self.training_data.length

    def __getitem__(self, index):
        item = self.training_data.get_data(index)
        # item.gen_mask(self.seq_len, mask_prob=self.mask_prob, mask_noise_prob=self.mask_noise_prob)

        return item.to_torch()
