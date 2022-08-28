import os
import torch

from pathlib import Path
from torch.utils.data import DataLoader
from dataset.dataset import BERTDataset
from model import TransInversion
from test.argparser import parse_input
from trainer.save_as_animation import save_as_animation


class AnimationData:
    def __init__(self, index, data):
        self.index = index
        self.fig = None
        self.frames = []
        self.data = data


def test_animation(base_path, test_file, indecies=[0, 10, 50, 100]):
    args = parse_input()
    animations = []

    print("Loading Test Dataset", test_file)
    test_dataset = BERTDataset(test_file)

    print("Building BERT model")
    args.hidden = 768
    generator = TransInversion(args.n_theta, test_dataset.seq_len, hidden=args.hidden,
                               n_layers=args.n_layers, n_decoder_layers=args.n_layers,
                               attn_heads=args.attn_heads, dropout=args.dropout)
    frames = None
    fig = None
    well_trans = test_dataset.training_data.well_trans
    train_data_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.num_workers)

    print("Training Start")
    for i, i_data in enumerate(train_data_loader):
        if i in indecies:
            animations.append(AnimationData(i, i_data))
            if len(animations) == len(indecies):
                break

    file_list = Path(base_path).glob('*.model')
    file_list = [str(item) for item in file_list]
    map_info2file = dict()
    for i, file_name in enumerate(file_list):
        model_name = os.path.basename(file_name)[:-6]
        infos = model_name.split("_")
        epoch = int(infos[1][2:])
        iteration = int(infos[2][4:])
        map_info2file[file_name] = (epoch, iteration)

    items = sorted(map_info2file.items(), key=lambda v: v[1], reverse=False)
    for i, (file_name, (epoch, iteration)) in enumerate(items):
        print(f"Loading model {i}: {file_name}...")
        model = torch.load(file_name)
        generator.load_state_dict(model.module.state_dict())

        for anim in animations:
            data = anim.data
            d_predict, well_predict1, well_predict2 = generator.forward(data["masked_d"], data["init_data"])
            anim.frames, anim.fig = save_as_animation(epoch, iteration,
                                                      data['d'], d_predict, data["masked_d"],
                                                      well_trans, data["init_data"],
                                                      data["well_data"],
                                                      well_predict1, well_predict2,
                                                      output_path=base_path,
                                                      frames=anim.frames,
                                                      index=anim.index,
                                                      fig=anim.fig)


if __name__ == '__main__':
    base_path = "E:/ShihuanLiu/code/trans-inversion/data_h768_b40"
    test_file = "../data/welldata_tests.mat"
    indecies = [0, 1, 100, 200, 300, 400, 500, 600, 700, 800]
    test_animation(base_path, test_file, indecies=indecies)
