from torch.utils.data import Dataset
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import torch

from typing import List

import csv

# === Source ===
# Sign Language classification dataset posted on
# Kaggle in 2017, by an unnamed author with username `tecperson`:
# https://www.kaggle.com/datamunge/sign-language-mnist
# Each sample is 1 x 1 x 28 x 28, and each label is a scalar.


# class for loading Sign Language dataset into PyTorch
class SignLanguageMNIST(Dataset):

    # letters in Latin alphabet range from 0 to 25
    # 9th letter (J) and 26th letter (Z) are motions: skip
    def get_label_mapping():
        mapping = list(range(25))
        mapping.pop(9)
        return mapping

    # takes in path and read each line
    def read_label_samples_from_csv(path: str):
        mapping = SignLanguageMNIST.get_label_mapping()
        labels, samples = [], []

        with open(path) as f:
            _ = next(f)  # skip header
            for line in csv.reader(f):

                # line = labels followed by pixel's value (0-255)
                label = int(line[0])
                labels.append(mapping.index(label))

                # convert to list of integers
                samples.append(list(map(int, line[1:])))

        return labels, samples

    def __init__(self,
                 path: str = "data/sign_mnist_train.csv",
                 mean: List[float] = [0.485],
                 std: List[float] = [0.229]):

        # read labels and samples
        labels, samples = SignLanguageMNIST.read_label_samples_from_csv(path)

        # reshape to make them numpy arrays instead of lists
        self._samples = np.array(samples, dtype=np.uint8).reshape((-1, 28, 28, 1))   # noqa
        self._labels = np.array(labels, dtype=np.uint8).reshape((-1, 1))

        # set up values from default
        self._mean = mean
        self._std = std

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        # noqa randomly zoom to increase robustness of model, and scale to 0-1 instead of 0-255
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(28, scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self._mean, std=self._std)])

        return {
            'image': transform(self._samples[idx]).float(),
            'label': torch.from_numpy(self._labels[idx]).float()
        }


# loads in data for train and test set
def get_train_test_loaders(batch_size=32):
    trainset = SignLanguageMNIST('data/sign_mnist_train.csv')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)    # noqa

    testset = SignLanguageMNIST('data/sign_mnist_test.csv')
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)     # noqa
    return trainloader, testloader


if __name__ == '__main__':
    loader, _ = get_train_test_loaders(2)
    print(next(iter(loader)))
