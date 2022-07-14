from torch.utils.data import DataLoader
from dataset.dataset import BERTDataset

train_file = 'D:/code_projects/matlab_projects/src/trans_gan_inversion/training_data.mat'
ds = BERTDataset(train_file)

data_loader = DataLoader(ds)
num_epoches = 100
for epoch in range(num_epoches):
    for i, data in enumerate(data_loader):
        print(data)
        exit(0)

