import h5py
import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset

from torchvision.transforms import *
import torchvision.transforms.functional as TF
from sklearn.metrics import roc_auc_score, roc_curve

BATCH_SIZE = 128
N_WORKERS = 14
H5_PATH = '/home/francesco/Desktop/carino/vaevictis/data_many_dist_fixed_step.h5'
GROUP = np.arange(1)
EPOCHES = 100
TRAIN = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PerceptionDataset(Dataset):
    def __init__(self, group, h5_path, transform=None, test=False):
        h5f = h5py.File(h5_path, 'r')

        self.Xs = {i: h5f['bag' + str(i) + '_x'] for i in group}
        self.Ys = {i: h5f['bag' + str(i) + '_y'] for i in group}
        self.lengths = {i: self.Xs[i].shape[0] for i in group}
        self.counts = {i: 0 for i in group}
        self.transform = transform
        self.test = test

        self.Y = []
        self.X = []

        self.length = 0

        for i in group:
            self.X += list(self.Xs[i][:])
            self.Y += list(self.Ys[i][:])

        for el in self.Xs.values():
            self.length += len(el)

    def __getitem__(self, item):
        g_len = self.length // len(self.X)

        g = len(self.X) - (self.length // (g_len + 1))

        x = self.X[item]
        y = self.Y[item]

        x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()

        x = x.permute(2, 0, 1)

        if self.transform is not None:
            x = self.transform(x)


        y[y > 0] = 1.0

        return x, y

    def __len__(self):
        return self.length

def conv_act_max(act='relu', *args, **kwargs):
    activation = nn.ModuleDict({
                        'relu': nn.ReLU(),
                        'lrelu': nn.LeakyReLU()
                })[act]

    return nn.Sequential(
        nn.Conv2d(*args, **kwargs, kernel_size=3, padding=1),
        activation,
        nn.MaxPool2d(kernel_size=2)
    )

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.encoder = nn.Sequential(
            *conv_act_max(in_channels=in_channels, out_channels=10),
            *conv_act_max(in_channels=10, out_channels=10),
            *conv_act_max(in_channels=10, out_channels=8),
        )

        self.decoder = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=8*8*10, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=65*5),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.flatten(1)
        x = self.decoder(x)

        return x
def get_dl(*args, **kwargs):
    ds = PerceptionDataset(*args, **kwargs)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=N_WORKERS, drop_last=True, shuffle=not kwargs['test'])

    return dl

if TRAIN: train_dl = get_dl(np.arange(2, 8), H5_PATH, test=False)

model = SimpleCNN().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002)

if TRAIN:
    # TRAIN
    for epoch in range(EPOCHES):
        tot_loss = torch.zeros(1).to(device)
        bar = tqdm.tqdm(enumerate(train_dl))

        for n_batch, (x, y) in bar:

            optimizer.zero_grad()

            x, y = x.to(device), y.to(device)

            mask = torch.ones(y.size()).to(device)
            mask[y == -1] = 0

            y_ = model(x)
            loss = criterion(y_ * mask, y * mask)

            loss.backward()

            optimizer.step()

            with torch.no_grad():
                tot_loss += loss

            bar.set_description('epoch={}, loss={:.4f}'.format(epoch, (tot_loss / (n_batch + 1)).cpu().item()))

    torch.save(model, './model.pt')

model  = torch.load('./model.pt')
test_dl = get_dl(np.arange(9, 11), H5_PATH, test=True)

# TESTa n
tot_loss = torch.zeros(1).to(device)
bar = tqdm.tqdm(enumerate(test_dl))
tot_auc = 0
i = 0

model.eval()
with torch.no_grad():
    for n_batch, (x, y) in bar:

        x, y = x.to(device), y.to(device)

        mask = torch.ones(y.size()).to(device)
        mask[y == -1] = 0

        y_ = model(x)

        y_ = y_.cpu().numpy()
        y = y.cpu().numpy()

        for y_1, y1 in zip(y_, y):
            indices = np.where(y1)

            y_1 = y_1[indices]
            y1 = y1[indices]
            try:
                auc = roc_auc_score(y1, y_1)
            except ValueError as e:
                auc = 0.5
            tot_auc += auc
            i += 1

print(i)
print(len(test_dl))

print('auc={:.4f}'.format((tot_auc / i)))
# print('epoch={}, loss={:.4f}'.format(epoch, (tot_loss / n_batch).cpu().item()))
#     y_
#
# loss =