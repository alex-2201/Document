import torch
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from digit_transform import RandomRotation, RandomShift

train_df = pd.read_csv('data/train.csv')
n_pixels = len(train_df.columns) - 1


class digit_data(Dataset):
    """Digit data set"""

    def __init__(self, file_path, n_pixels,
                 transform=transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                               transforms.Normalize(mean=(0.5,), std=(0.5,))]),
                 ):

        df = pd.read_csv(file_path)

        if len(df.columns) == n_pixels:
            # test data
            self.X = df.values.reshape((-1, 28, 28)).astype(np.uint8)[:, :, :, None]
            self.y = None
        else:
            # training data
            self.X = df.iloc[:, 1:].values.reshape((-1, 28, 28)).astype(np.uint8)[:, :, :, None]
            self.y = torch.from_numpy(df.iloc[:, 0].values)

        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.transform(self.X[idx]), self.y[idx]
        else:
            return self.transform(self.X[idx])


batch_size = 64

train_dataset = digit_data('data/train.csv', transform=transforms.Compose(
    [transforms.ToPILImage(), RandomRotation(degrees=20), RandomShift(3),
     transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]), n_pixels=n_pixels)
test_dataset = digit_data('data/test.csv', n_pixels=n_pixels)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, shuffle=False)
