from torch.utils.data import DataLoader

from dataset import BERTDataset
from model import BERT, TransInversion
from trainer import BERTTrainer

batch_size = 8
num_workers = 0
hidden = 192
n_layers = 6
attn_heads = 8
lr = 1e-3
adam_beta1 = 0.9
adam_beta2 = 0.999
weight_decay = 0.01
with_cuda = True
cuda_devices = [0]
log_freq = 10
epochs = 300
miu = 0.01
loss_save_path = "../data/convergence.html"
output_path = "../data/trans_gan_inversion.model"

train_file = 'D:/code_projects/matlab_projects/src/trans_gan_inversion/training_data.mat'
test_file = 'D:/code_projects/matlab_projects/src/trans_gan_inversion/test_data.mat'
# test_file = None

print("Loading Train Dataset", train_file)
train_dataset = BERTDataset(train_file)

print("Loading Test Dataset", test_file)
test_dataset = BERTDataset(test_file) if test_file is not None else None

print("Creating Dataloader")
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers) \
    if test_dataset is not None else None

print("Building BERT model")
bert = BERT(angle_num=train_dataset.angle_num, hidden=hidden, n_layers=n_layers, attn_heads=attn_heads)
trans_inversion = TransInversion(bert, n_layers=n_layers)

print("Creating BERT Trainer")
trainer = BERTTrainer(bert, train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                      lr=lr, betas=(adam_beta1, adam_beta2), weight_decay=weight_decay,
                      miu=miu, loss_save_path=loss_save_path,
                      with_cuda=with_cuda, cuda_devices=cuda_devices, log_freq=log_freq)

print("Training Start")
for epoch in range(epochs):
    trainer.train(epoch)
    trainer.save(epoch, output_path)

    if test_data_loader is not None:
        trainer.test(epoch)
