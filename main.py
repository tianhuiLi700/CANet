import torch
from fandata import FanDataset
import argparse
from torch.utils.data import DataLoader
import numpy as np
import random
import time
import os
from build_model import build_models
from engine import train_one_epoch, eval_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('Fan', add_help=False)
    parser.add_argument('--model_name', default='ours', type=str)

    # about lr
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--save_file', default='./outputs', type=str)

    # parser.add_argument('--lr_drop', default=200, type=int)

    parser.add_argument('--dataset_file', default='new_data.txt')
    parser.add_argument('--device', default='cuda:2', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--sample_step', default=3, type=int)
    parser.add_argument('--seq_len', default=50, type=int)

    parser.add_argument('--num_para', default=4, type=int)
    parser.add_argument('--cv_num', default=5, type=int)

    parser.add_argument('--hidden_dim', default=512, type=int)
    parser.add_argument('--dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)
    # lstm
    parser.add_argument('--lstm_layers', default=6, type=int)


    return parser


def main(args):

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)




    # MonteCarloCv 5
    data_ancher = [random.random() * 0.2 + 0.64 for x in range(args.cv_num)]
    ancher_dir = [(random.random() < 0.5) * 2 - 1  for x in range(args.cv_num)]
    split_point = [data_ancher[i] * ancher_dir[i] for i in range(args.cv_num)]
    max_data_len = len(np.loadtxt(args.dataset_file, delimiter=' '))
    s_e_points = split_data(split_point, train_len=0.64, test_len=0.16)
    s_e_points = [{k: int(v * (max_data_len - 1)) for k, v in ps.items()} for ps in s_e_points]
    all_mes = 0
    for cv_i in range(args.cv_num):

        model = build_models(args, args.model_name)
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        train_data = FanDataset(args.dataset_file, args.sample_step, args.seq_len,
                                start_idx=s_e_points[cv_i]['train_s'], end_idx=s_e_points[cv_i]['train_e'])
        train_loder = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        test_data = FanDataset(args.dataset_file, args.sample_step, args.seq_len,
                               start_idx=s_e_points[cv_i]['test_s'], end_idx=s_e_points[cv_i]['test_e'])
        test_loder = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

        log = {}
        best_acc = 0
        best_epoch = 0
        for i in range(args.epochs):
            loss = train_one_epoch(model, train_loder, device, i, optimizer)
            log['train_loss'] = loss
            accuracy, precision, recall, f1, report = eval_one_epoch(model, test_loder, device, i, optimizer)
            log['val_acc'] = accuracy

            if log['val_acc'] >= best_acc:
                best_acc = log['val_acc']
                best_epoch = i
                torch.save(model.state_dict(),
                           os.path.join(args.save_file, "cv_{}_{}_{}_5.pth".format(cv_i, args.model_name, 'best')))
            with open('./out_imf/result_{}_all_5.txt'.format(args.model_name), 'a') as f:
                f.writelines("Cv:{} epoch:{} train_loss:{:.5f} val_acc:{:.5f}"
                             " accuracy:{:.5f} precision:{:.5f} recall:{:.5f} f1:{:.5f}"
                             " best_epoch:{} best_acc:{:.5f}\n".format(cv_i, i, log['train_loss'], log['val_acc'],
                                                                        accuracy, precision, recall, f1,
                                                                        best_epoch, best_acc))
            f.close()
            with open('./out_imf/result_{}_report_5.txt'.format(args.model_name), 'a') as f:
                f.writelines("Cv:{} epoch:{} train_loss:{:.5f} val_acc:{:.5f}"
                             " accuracy:{:.5f} precision:{:.5f} recall:{:.5f} "
                             "f1:{:.5f}\n".format(cv_i, i, log['train_loss'], log['val_acc'],
                                                accuracy, precision, recall, f1) + report)
        all_mes += best_acc
        print("Cv:{:.5f}".format(all_mes/5))
        with open('result_{}.txt'.format(args.model_name), 'a') as f:
            f.writelines("Cv result: {:.5f}".format(all_mes/5))
        f.close()




def split_data(point, train_len=0.64, test_len=0.16):
    out = []
    for p in point:
        if p < 0:
            p = 1 + p
            test_start = p - 0.16 if (p - 0.16) >= 0 else 0
            test_end = train_start = p
            train_end = p + 0.64 if (p + 0.64) <= 1 else 1
        else:
            test_start = train_end = p
            test_end = p + 0.16 if (p + 0.16) <= 1 else 1
            train_start = p - 0.64 if (p - 0.64) >= 0 else 0
        out.append({'train_s': train_start, 'train_e': train_end, 'test_s': test_start, 'test_e': test_end})
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

