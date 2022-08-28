#!/usr/bin/env python

import argparse
import sys
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import random

from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from training import train
from model import VGG16
from eval import eval_model

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True,
                    help='The path of the input dataset')
parser.add_argument('--model', type=str, required=True,
                    help='The model to train, with initialised weights, serialised with torch.save()')
parser.add_argument('--first_round', action='store_true',
                    help='Is this the first round or not?')
parser.add_argument('--epochs_per_round', type=int, required=True,
                    help='Number of epochs per round')

args = parser.parse_args(sys.argv[1:])

data_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
  transforms.Resize(32),
  transforms.ToTensor(),
])

train_set = torchvision.datasets.MNIST(root=args.dataset, download=False, train=True, transform=data_transform)
test_set = torchvision.datasets.MNIST(root=args.dataset, download=False, train=False, transform=test_transform)
   
num_train = len(train_set)
num_test = len(test_set)

x_train, y_train=train_set.data,train_set.targets

if len(x_train.shape)==3:
  x_train=x_train.unsqueeze(1)

train_idx = list(range(num_train))

test_idx = list(range(num_test))

random.shuffle(train_idx)

val_frac = 0.1
num_val = int(num_train * val_frac) 
num_train = num_train - num_val

val_idx = train_idx[num_train:]
train_idx = train_idx[:num_train]

val_set = Subset(train_set, val_idx)
train_set = Subset(train_set, train_idx)

train_loader = DataLoader(train_set, batch_size=64, num_workers=0, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_set,   batch_size=64, num_workers=0, shuffle=False, drop_last=False)
test_loader  = DataLoader(test_set,  batch_size=64, num_workers=0, shuffle=False, drop_last=False)

loaders = {"train": train_loader,
           "val": val_loader,
           "test": test_loader}

if torch.cuda.is_available():
    dev = torch.device('cuda')
else:
    dev = torch.device('cpu')

PATH = args.model

model = VGG16(10)
model.block_1[1] = nn.GroupNorm(32, 64)
model.block_1[4] = nn.GroupNorm(32, 64)
model.block_2[1] = nn.GroupNorm(32, 128)
model.block_2[4] = nn.GroupNorm(32, 128)
model.block_3[1] = nn.GroupNorm(32, 256)
model.block_3[4] = nn.GroupNorm(32, 256)
model.block_3[7] = nn.GroupNorm(32, 256)
model.block_4[1] = nn.GroupNorm(32, 512)
model.block_4[4] = nn.GroupNorm(32, 512)
model.block_4[7] = nn.GroupNorm(32, 512)
model.load_state_dict(torch.load(PATH, map_location=dev))

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

if args.first_round:
    print("TRAINING LOCAL MODEL ON MNIST")
    train(model, loaders, optimizer, criterion, epochs=args.epochs_per_round, dev=dev)    
    torch.save(model.state_dict(), "state_dict_model.pt")
    
else:
    print("TEST MODEL ON MNIST:")
    eval_model(model, test_loader)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    print("TRAINING LOCAL MODEL ON MNIST")
    train(model, loaders, optimizer, criterion, epochs=args.epochs_per_round, dev=dev)
    torch.save(model.state_dict(), "state_dict_model.pt")
