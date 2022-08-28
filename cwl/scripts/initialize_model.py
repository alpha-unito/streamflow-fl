#!/usr/bin/env python

import torch
import torch.nn as nn
from model import VGG16

net = VGG16(10)

net.block_1[1] = nn.GroupNorm(32, 64)
net.block_1[4] = nn.GroupNorm(32, 64)
net.block_2[1] = nn.GroupNorm(32, 128)
net.block_2[4] = nn.GroupNorm(32, 128)
net.block_3[1] = nn.GroupNorm(32, 256)
net.block_3[4] = nn.GroupNorm(32, 256)
net.block_3[7] = nn.GroupNorm(32, 256)
net.block_4[1] = nn.GroupNorm(32, 512)
net.block_4[4] = nn.GroupNorm(32, 512)
net.block_4[7] = nn.GroupNorm(32, 512)

torch.save(net.state_dict(), "state_dict_model.pt")
