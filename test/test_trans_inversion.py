from torch.utils.data import DataLoader
from dataset import BERTDataset
from model import BERT, TransInversion

train_file = 'D:/code_projects/matlab_projects/src/trans_gan_inversion/training_data.mat'
ds = BERTDataset(train_file)
model = BERT(angle_num=ds.angle_num)
trans_inversion = TransInversion(model)

data_loader = DataLoader(ds)
num_epoches = 100
for epoch in range(num_epoches):
    for i, data in enumerate(data_loader):
        d_new, well_new = trans_inversion(data['masked_d'], data['init_data'])
        print(d_new.shape, well_new.shape)
        exit(0)
