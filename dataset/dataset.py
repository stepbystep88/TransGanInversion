import mat73
import torch
from scipy import signal
from torch.utils.data import Dataset
import numpy as np
import scipy.io as scio
from sklearn.preprocessing import MinMaxScaler

from seismic.synthesis import Synthesis


class TrainingItem:
    def __init__(self, orig_well_data, well_data, init_data):
        self.orig_well_data = orig_well_data
        self.well_data = well_data
        self.init_data = init_data

    def get(self):
        output = {
            "orig_well_data": self.orig_well_data,
            "well_data": self.well_data,
            "init_data": self.init_data
        }

        return output


class TrainingData:
    """
    训练数据，从mat文件读取
    """
    def __init__(self, training_data_path, need_normalize=True, well_trans=None):
        try:
            data = scio.loadmat(training_data_path)
        except:
            data = mat73.loadmat(training_data_path)

        self.data = data

        self.well_data = data['wellData'][:, :, 1:].astype(np.float32)
        self.orig_well_data = self.well_data

        b, a = signal.butter(8, 0.04, 'lowpass')
        self.init_data = signal.filtfilt(b, a, self.well_data, axis=0)  # data为要过滤的信号
        self.well_trans = well_trans
        if need_normalize:
            self.normalize()

    def normalize(self):
        self.well_trans = MinMaxScaler(feature_range=(-1, 1))
        seq_len = self.well_data.shape[0]
        self.well_data = np.reshape(self.well_trans.fit_transform(np.reshape(self.well_data, (-1, 3))), (seq_len, -1, 3))
        self.init_data = np.reshape(self.well_trans.transform(np.reshape(self.init_data, (-1, 3))), (seq_len, -1, 3))

        self.well_data = self.well_data.astype(np.float32)
        self.init_data = self.init_data.astype(np.float32)

    def get_data(self, index) -> TrainingItem:
        orig_well_data = self.orig_well_data[:, index, :]
        well_data = self.well_data[:, index, :]
        init_data = self.init_data[:, index, :]
        item = TrainingItem(orig_well_data, well_data, init_data)

        return item

    @property
    def seq_len(self):
        return self.well_data.shape[0]

    @property
    def length(self):
        return self.well_data.shape[1]


class BERTDataset(Dataset):
    def __init__(self, training_data_path, mask_prob=0.2, mask_noise_prob=0.5, dt=2,
                 freq_range=(28, 60), snr_range=(2, 10)):

        self.training_data = TrainingData(training_data_path)
        self.seq_len = self.training_data.seq_len
        self.mask_prob = mask_prob
        self.mask_noise_prob = mask_noise_prob
        self.synthesiser = Synthesis(self.seq_len)
        self.dt = self.dt
        self.freq_range = freq_range
        self.snr_range = snr_range

    def __len__(self):
        return self.training_data.length

    def __getitem__(self, index):
        item = self.training_data.get_data(index)
        data = item.get()
        data = self.gen_mask(data)
        return {key: torch.from_numpy(value) for key, value in data.items()}

    @staticmethod
    def add_noise(x_volts, target_snr_dbs):
        y_volts = np.zeros_like(x_volts)
        for k, target_snr_db in enumerate(target_snr_dbs):
            x_watts = x_volts[k, :, :] ** 2
            # Calculate signal power and convert to dB
            sig_avg_watts = np.mean(x_watts)
            sig_avg_db = 10 * np.log10(sig_avg_watts)
            # Calculate noise according to [2] then convert to watts
            noise_avg_db = sig_avg_db - target_snr_db
            noise_avg_watts = 10 ** (noise_avg_db / 10)
            # Generate an sample of white noise
            mean_noise = 0
            noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), x_watts.shape)
            # Noise up the original signal
            y_volts[k, :, :] = x_volts[k, :, :] + noise_volts

        return y_volts

    def gen_mask(self, data):
        well_data = data["orig_well_data"].numpy()
        batch_size, seq_len, _ = data["well_data"].shape
        freqs = np.random.uniform(self.freq_range[0], self.freq_range[1], (batch_size,))
        snrs = np.random.uniform(self.snr_range[0], self.snr_range[1], (batch_size,))

        d = self.synthesiser.gen_pre_angle_data(
            well_data[:, :, 0].squeeze(),
            well_data[:, :, 1].squeeze(),
            well_data[:, :, 1].squeeze(),
            freqs, self.dt
        )
        d = np.array(d)
        d_noise = self.add_noise(d, snrs)

        n_mask = int((seq_len - 1) * self.mask_prob)
        masked_index = np.random.randint(0, seq_len - 1, n_mask).astype(np.int64)
        masked_d = d_noise.copy()
        masked_d[:, masked_index, :] = 0

        data["d"] = d
        data["d_noise"] = d_noise
        data["masked_d"] = masked_d
        data["masked_index"] = np.repeat(masked_index, batch_size, axis=0)

        return data
