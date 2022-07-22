import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset.gen_mask import gen_mask
from trainer.optim_schedule import ScheduledOptim

import tqdm
import os
import pandas as pd
import plotly.express as px

from trainer.utils import save_cmp_as_html


class BERTTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, generator, discriminator, hidden, attn_heads, well_trans,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10,
                 mask_prob=0.2, mask_noise_prob=0.5,
                 miu=(10, 1, 0.1), loss_save_path=None):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        self.well_trans = well_trans

        # Initialize the BERT Language Model, with BERT model
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.generator = nn.DataParallel(self.generator, device_ids=cuda_devices)
            self.discriminator = nn.DataParallel(self.discriminator, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.attn_heads = attn_heads
        self.mask_prob = mask_prob
        self.mask_noise_prob = mask_noise_prob

        # Setting the Adam optimizer with hyper-param
        self.adam_G = Adam(self.generator.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_G = ScheduledOptim(self.adam_G, hidden, n_warmup_steps=warmup_steps)

        self.adam_D = Adam(self.discriminator.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_D = ScheduledOptim(self.adam_D, hidden, n_warmup_steps=warmup_steps)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.mse_criterion = nn.MSELoss()
        self.bce_criterion = nn.BCELoss()

        self.log_freq = log_freq
        self.miu = miu
        self.loss_save_path = loss_save_path
        self.train_loss_info = []
        self.test_loss_info = []

        print("Total Parameters of generator:", sum([p.nelement() for p in self.generator.parameters()]))
        print("Total Parameters of discriminator:", sum([p.nelement() for p in self.discriminator.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    @staticmethod
    def batched_index_select(t, dim, inds):
        dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
        out = t.gather(dim, dummy)  # b x e x f
        return out

    def train_D(self, train, data):
        well_label = data["well_data"]
        batch_szie = well_label.shape[0]

        with torch.no_grad():
            _, _, well_predict = self.generator.forward(data["masked_d"], data["init_data"])

        # Forward with real data
        real_label = torch.ones(batch_szie).to(self.device)
        real_logit = self.discriminator(well_label).flatten()
        lossD_real = self.bce_criterion(real_logit, real_label)

        if train:
            self.optim_D.zero_grad()
            lossD_real.backward()

        # Forward with fake data
        fake_label = torch.zeros(batch_szie).to(self.device)
        fake_logit = self.discriminator(well_predict).flatten()
        lossD_fake = self.bce_criterion(fake_logit, fake_label)

        if train:
            lossD_fake.backward()
            self.optim_D.step_and_update_lr()

        lossD = lossD_real + lossD_fake
        return lossD, lossD_real, lossD_fake

    def mse_loss(self, x, y):
        residual = x - y
        return torch.sum(residual * residual)

    def train_G(self, train, data):
        well_label = data["well_data"]
        batch_szie = well_label.shape[0]

        d_predict, well_predict1, well_predict2 = self.generator.forward(data["masked_d"], data["init_data"])
        masked_d = self.batched_index_select(data["d"], 1, data["masked_index"])
        masked_d_predict = self.batched_index_select(d_predict, 1, data["masked_index"])
        #
        d_loss = self.mse_loss(masked_d, masked_d_predict)
        well_loss1 = self.mse_loss(well_label, well_predict1)
        well_loss2 = self.mse_loss(well_label, well_predict2)

        # loss of discriminator
        # real_label = torch.ones(batch_szie).to(self.device)
        # real_logit = self.discriminator(well_predict2).flatten()
        # lossG_D = self.bce_criterion(real_logit, real_label)
        lossG_D = d_loss

        lossG = self.miu[0] * d_loss + self.miu[1] * (well_loss1 + well_loss2) + self.miu[2] * lossG_D
        if train:
            self.optim_G.zero_grad()
            lossG.backward()
            self.optim_G.step_and_update_lr()
            
        # return lossG, d_loss, well_loss, lossG_D
        loss = (lossG, lossG_D, d_loss, well_loss1, well_loss2)
        predict = d_predict, well_predict1, well_predict2
        return predict, loss

    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        for i, orig_data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = gen_mask(orig_data, self.attn_heads, mask_prob=self.mask_prob, mask_noise_prob=self.mask_noise_prob)
            data = {key: value.to(self.device) for key, value in data.items()}

            # lossD, lossD_real, lossD_fake = self.train_D(train, data)
            well_label = data["well_data"]
            predict, loss = self.train_G(train, data)
            lossG, lossG_D, d_loss, well_loss1, well_loss2 = loss
            d_predict, well_predict1, well_predict2 = predict

            well_predict1 = well_predict1.cpu().detach().numpy()
            well_predict2 = well_predict2.cpu().detach().numpy()
            well_label = well_label.cpu().detach().numpy()

            post_fix = {
                "type": str_code,
                "epoch": epoch,
                "step": epoch * len(data_iter) + i,
                "iter": i,
                # "lossD": lossD.item(),
                # "lossD_real": lossD_real.item(),
                # "lossD_fake": lossD_fake.item(),
                "lossG": lossG.item(),
                "d_loss": d_loss.item(),
                "well_loss1": well_loss1.item(),
                "well_loss2": well_loss2.item(),
                "min_predict1": well_predict1[:, :, 0].min(),
                "max_predict1": well_predict1[:, :, 0].max(),
                "min_predict2": well_predict2[:, :, 0].min(),
                "max_predict2": well_predict2[:, :, 0].max(),
                "min_label": well_label[:, :, 0].min(),
                "max_label": well_label[:, :, 0].max(),
                "lossG_D": lossG_D.item()
            }

            if train:
                self.train_loss_info.append(post_fix)
            else:
                self.test_loss_info.append(post_fix)

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
                if self.loss_save_path:
                    self.save_loss_as_html(len(data_iter))

            if i % 30 == 0:
                save_cmp_as_html(data["d"], d_predict, data["masked_d"], self.well_trans, data["init_data"],
                                 well_label, well_predict1, well_predict2,
                                 output_path=f"{os.path.dirname(os.path.dirname(__file__))}/data/")

        print("EP%d_%s" % (epoch, str_code))

    def save_loss_as_html(self, n):
        """
        保存收敛曲线
        :param n:
        """
        save_path = self.loss_save_path
        df = pd.DataFrame(self.train_loss_info)

        if self.test_loss_info:
            df2 = pd.DataFrame(self.test_loss_info)
            df = pd.concat([df, df2])
        for loss_name in ["well_loss1", "well_loss2", "lossG", "d_loss", "lossG_D"]:
            fig = px.line(df, x="step", y=loss_name, color='type', log_y=True)
            fig.write_html(save_path.replace(".html", f"_{loss_name}.html"))

    def save(self, epoch, file_path):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        self.save_model(self.generator, epoch, f"{file_path}/generator.model.ep{epoch}")
        self.save_model(self.discriminator, epoch, f"{file_path}/discriminator.model.ep{epoch}")

    def save_model(self, model, epoch, output_path):
        torch.save(model.cpu(), output_path)
        model.to(self.device)

        print("EP:%d Model Saved on:" % epoch, output_path)
