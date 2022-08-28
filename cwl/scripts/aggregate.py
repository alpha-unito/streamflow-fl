#!/usr/bin/env python

import argparse
import sys
import torch

from torch import nn

from model import VGG16
from aggregation import init_params

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, required=True, action='append',
                    help='The trained models, serialised with torch.save()')

args = parser.parse_args(sys.argv[1:])

if torch.cuda.is_available():
    dev = torch.device('cuda')
else:
    dev = torch.device('cpu')

nets = []

for path in args.model:
    sd = torch.load(path, map_location=dev)
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
    net.load_state_dict(sd)
    nets.append(net)

aggregated_model = VGG16(10)
aggregated_model.block_1[1] = nn.GroupNorm(32, 64)
aggregated_model.block_1[4] = nn.GroupNorm(32, 64)
aggregated_model.block_2[1] = nn.GroupNorm(32, 128)
aggregated_model.block_2[4] = nn.GroupNorm(32, 128)
aggregated_model.block_3[1] = nn.GroupNorm(32, 256)
aggregated_model.block_3[4] = nn.GroupNorm(32, 256)
aggregated_model.block_3[7] = nn.GroupNorm(32, 256)
aggregated_model.block_4[1] = nn.GroupNorm(32, 512)
aggregated_model.block_4[4] = nn.GroupNorm(32, 512)
aggregated_model.block_4[7] = nn.GroupNorm(32, 512)

init_params(aggregated_model, nets)
torch.save(aggregated_model.state_dict(), "state_dict_model.pt")

