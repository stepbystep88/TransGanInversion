from torch.utils.data import DataLoader

from dataset.dataset import BERTDataset
from model import BERT

train_file = 'D:/code_projects/matlab_projects/src/trans_gan_inversion/training_data.mat'
ds = BERTDataset(train_file)
model = BERT(angle_num=ds.angle_num)

data_loader = DataLoader(ds)
num_epoches = 100
for epoch in range(num_epoches):
    for i, data in enumerate(data_loader):
        output = model(data['masked_d'])
        print(output)
        exit(0)
