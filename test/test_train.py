import os

from torch.utils.data import DataLoader

from dataset.dataset import BERTDataset
from model import TransInversion, Discriminator
from test.argparser import parse_input
from trainer import BERTTrainer


def train():
    args = parse_input()

    loss_save_path = f"{args.output_path}/convergence.html"
    output_model_path = f"{args.output_path}"
    os.makedirs(output_model_path, exist_ok=True)

    print("Loading Train Dataset", args.train_file)
    train_dataset = BERTDataset(args.train_file)

    print("Loading Test Dataset", args.test_file)
    test_dataset = BERTDataset(args.test_file) if args.test_file is not None else None

    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers) \
        if test_dataset is not None else None

    print("Building BERT model")
    generator = TransInversion(train_dataset.angle_num, train_dataset.seq_len, hidden=args.hidden,
                               n_layers=args.n_layers, n_decoder_layers=args.n_layers,
                               attn_heads=args.attn_heads, dropout=args.dropout)

    discriminator = Discriminator(hidden=args.hidden, n_layers=args.n_layers, attn_heads=args.attn_heads,
                                  dropout=args.dropout)

    print("Creating BERT Trainer")
    trainer = BERTTrainer(generator, discriminator, args.hidden, args.attn_heads,
                          train_dataset.training_data.well_trans,
                          train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                          lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.weight_decay,
                          with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq,
                          loss_save_path=loss_save_path, miu=(args.miu0, args.miu1, args.miu2))

    print("Training Start")
    for epoch in range(args.epochs):
        trainer.train(epoch)
        trainer.save(epoch, output_model_path)

        if test_data_loader is not None:
            trainer.test(epoch)


if __name__ == '__main__':
    train()