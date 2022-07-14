import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from model import TransInversion, BERT
from trainer.optim_schedule import ScheduledOptim

import tqdm
import pandas as pd
import plotly.express as px


class BERTTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, bert: BERT, train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10, miu=0.01, loss_save_path=None):
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

        # This BERT model will be saved every epoch
        self.bert = bert
        # Initialize the BERT Language Model, with BERT model
        self.model = TransInversion(bert).to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.MSELoss()

        self.log_freq = log_freq
        self.miu = miu
        self.loss_save_path = loss_save_path
        self.train_loss_info = []
        self.test_loss_info = []

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    @staticmethod
    def batched_index_select(t, dim, inds):
        dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
        out = t.gather(dim, dummy)  # b x e x f
        return out

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

        avg_loss = 0.0
        avg_d_loss = 0.0
        avg_well_loss = 0.0

        for i, orig_data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in orig_data.items()}

            # 1. forward the next_sentence_prediction and masked_lm model
            new_d, new_well = self.model.forward(data["masked_d"], data["init_data"])
            masked_index = data["masked_index"].to(torch.long)
            masked_d = self.batched_index_select(data["d"], 1, masked_index)
            masked_new_d = self.batched_index_select(new_d, 1, masked_index)

            # 2-1. NLL(negative log likelihood) loss of is_next classification result
            d_loss = self.criterion(masked_d, masked_new_d)

            # 2-2. NLLLoss of predicting masked token word
            well_loss = self.criterion(new_well, data["well_data"])

            # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
            loss = d_loss + self.miu * well_loss

            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            avg_loss += loss.item()
            avg_d_loss += d_loss.item()
            avg_well_loss += well_loss.item()

            precision = 4
            post_fix = {
                "type": str_code,
                "epoch": epoch,
                "iter": i,
                "avg_loss": round(avg_loss / (i + 1), precision),
                "avg_d_loss": round(avg_d_loss / (i + 1), precision),
                "avg_well_loss": round(avg_well_loss / (i + 1), precision),
                "loss": round(loss.item(), precision),
                "d_loss": round(d_loss.item(), precision),
                "well_loss": round(well_loss.item(), precision)
            }
            if train:
                self.train_loss_info.append(post_fix)
            else:
                self.test_loss_info.append(post_fix)

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
                if self.loss_save_path:
                    self.save_loss_as_html(len(data_iter))

        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter))

    def save_loss_as_html(self, n):
        """
        保存收敛曲线
        :param n:
        """
        save_path = self.loss_save_path
        df = pd.DataFrame(self.train_loss_info)
        df['step'] = n * df['epoch'] + df['iter']

        if self.test_loss_info:
            df2 = pd.DataFrame(self.test_loss_info)
            df2['step'] = n * df2['epoch'] + df['iter']
            df = pd.concat([df, df2])
        for loss_name in ["avg_loss", "avg_d_loss", "avg_well_loss", "loss", "d_loss", "well_loss"]:
            fig = px.line(df, x="step", y=loss_name, color='type', log_y=True)
            fig.write_html(save_path.replace(".html", f"_{loss_name}.html"))

    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.bert.cpu(), output_path)
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
