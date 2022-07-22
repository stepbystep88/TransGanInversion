import argparse


def parse_input():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--train_file", type=str, help="train dataset for train bert",
                        default='D:/code_projects/matlab_projects/src/trans_gan_inversion/training_data.mat')
    parser.add_argument("-t", "--test_file", type=str, help="test set for evaluate train set",
                        default=None)
    # D:/code_projects/matlab_projects/src/trans_gan_inversion/test_data.mat'
    parser.add_argument("-o", "--output_path", default='../data', type=str, help="")

    parser.add_argument("-hs", "--hidden", type=int, default=384, help="hidden size of transformer model")
    parser.add_argument("-l", "--n_layers", type=int, default=6, help="number of layers")
    parser.add_argument("-ld", "--n_decoder_layers", type=int, default=6, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=6, help="number of attention heads")

    parser.add_argument("-b", "--batch_size", type=int, default=16, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=0, help="dataloader worker size")

    parser.add_argument("--with_cuda", type=int, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n iter: setting n")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0], help="CUDA device ids")

    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate of adam")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    parser.add_argument("--miu0", type=float, default=1, help="miu0")
    parser.add_argument("--miu1", type=float, default=1, help="miu1")
    parser.add_argument("--miu2", type=float, default=0.1, help="miu2")

    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")

    args = parser.parse_args()

    print(args)
    
    return args