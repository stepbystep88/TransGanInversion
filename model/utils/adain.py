import torch
from torch import nn


class AdaIn(nn.Module):
    def forward(self, x, z):
        var_z, mean_z = torch.std_mean(z, dim=1, keepdim=True)
        var_x, mean_x = torch.std_mean(x, dim=1, keepdim=True)

        return var_z * (x - mean_x) / (var_x + 1e-10) + mean_z


class AdaInLinear(nn.Module):
    def __init__(self, hidden, out_channel):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(out_channel, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_channel)
        )
        self.adain = AdaIn()

    def forward(self, x, z):
        x = self.adain(x, z)
        return self.linear(x)

 # python .\test_train.py -b=32 --lr=1e-3 -w=4
# D:/code_projects/matlab_projects/src/trans_gan_inversion