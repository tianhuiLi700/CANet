from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import numpy as np
import torch


class FanDataset(Dataset):
    def __init__(self, file, step, seq_len, start_idx, end_idx):
        super(FanDataset, self).__init__()
        self.data = np.loadtxt(file, delimiter=' ')[start_idx: end_idx]
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.step = step
        self.seq_len = seq_len
        self.datalen = len(self.data) - (seq_len - 1) * step - 1

    def __len__(self):
        return self.datalen


    def sample(self, start, end):
        data = []
        for i in range(start, end + 1, self.step):
            data.append(self.data[i])
        return data

    def __getitem__(self, idx):
        start = idx
        end = start + self.step * (self.seq_len - 1)
        data = np.asarray(self.sample(start, end))
        data = self.transform(data)[0]
        seq = data[:, 1:5]
        pred = data[:, 5:]
        cls = torch.sum(pred, dim=0) > 0
        cls = cls.int()

        target = {"pred": pred, 'err': cls}

        return seq, target


