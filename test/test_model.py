import argparse
import os

import torch
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader

import numpy as np
from dataset.dataset import BERTDataset
from model import TransInversion, Discriminator
from test.argparser import parse_input
from trainer import BERTTrainer
import plotly.graph_objects as go


def load_and_cmp():
    args = parse_input()
    epoch = 7
    output_path = f"{args.output_path}"
    model_path = f"{output_path}/generator.model.ep{epoch}"

    print("Loading Train Dataset", args.train_file)
    train_dataset = BERTDataset(args.train_file)

    print("Loading Test Dataset", args.test_file)
    test_dataset = BERTDataset(args.test_file) if args.test_file is not None else None

    print("Building BERT model")
    # generator = TransInversion(train_dataset.angle_num, hidden=args.hidden,
    #                            n_layers=args.n_layers, n_decoder_layers=args.n_layers,
    #                            attn_heads=args.attn_heads, dropout=args.dropout)

    generator = torch.load(model_path)
    # generator.load_state_dict(model['state_dict'])

    well_trans = train_dataset.training_data.well_trans
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    print("Training Start")
    for i, data in enumerate(train_data_loader):
        # d_predict, well_predict = generator.forward(data["d"], data["init_data"], None)
        well_predict = generator.forward(data["d"])

        well_label = data["well_data"].cpu().detach().numpy()
        well_init = data["init_data"].cpu().detach().numpy()
        well_predict = well_predict.cpu().detach().numpy()
        batch_size, seq_len, _ = well_label.shape

        well_label = np.reshape(well_trans.inverse_transform(np.reshape(well_label, (-1, 3))), (batch_size, -1, 3))
        well_init = np.reshape(well_trans.inverse_transform(np.reshape(well_init, (-1, 3))), (batch_size, -1, 3))
        well_predict = np.reshape(well_trans.inverse_transform(np.reshape(well_predict, (-1, 3))), (batch_size, -1, 3))

        fig = make_subplots(rows=1, cols=3)

        y = list(range(seq_len))

        for k in range(3):
            fig.add_trace(
                go.Scatter(x=well_label[0, :, k], y=y, marker=dict(color="blue")),
                row=1, col=k+1
            )

            fig.add_trace(
                go.Scatter(x=well_init[0, :, k], y=y, marker=dict(color="green")),
                row=1, col=k+1
            )

            fig.add_trace(
                go.Scatter(x=well_predict[0, :, k], y=y, marker=dict(color="red")),
                row=1, col=k+1
            )

        fig.write_html(f"{output_path}/cmp_real_predict.html")

        exit(0)


if __name__ == '__main__':
    load_and_cmp()
