import numpy as np
import torch


def gen_mask(data, attn_heads, mask_prob=0.2, mask_noise_prob=0.5):
    d_noise = data["d_noise"]
    batch_size, seq_len, _ = data["well_data"].shape
    n_mask = int((seq_len - 1) * mask_prob)
    masked_index = np.random.randint(0, seq_len - 1, n_mask).astype(np.int64)

    mask = np.ones((seq_len, seq_len))
    mask[:, masked_index] = 0
    mask[masked_index, :] = 0

    masked_d = data["d"].clone()
    masked_d[:, masked_index, :] = 0
    n_mask_noise = int(n_mask * mask_noise_prob)
    noise_index = np.random.choice(masked_index, n_mask_noise, replace=True)
    masked_d[:, noise_index, :] = d_noise[:, noise_index, :]

    data["masked_d"] = masked_d
    data["mask"] = torch.tensor(mask).unsqueeze(0).repeat(batch_size, attn_heads, 1, 1)
    data["masked_index"] = torch.tensor(masked_index).repeat(batch_size, 1)

    return data