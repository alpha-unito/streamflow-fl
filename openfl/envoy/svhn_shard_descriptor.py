# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SVHM Shard Descriptor."""

import logging
import os
from typing import List
import scipy.io as sio
import torchvision
from torchvision import transforms
import cv2

import numpy as np
import requests

from openfl.interface.interactive_api.shard_descriptor import ShardDataset
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor

logger = logging.getLogger(__name__)


class SVHNShardDataset(ShardDataset):
    """SVHN Shard dataset class."""

    def __init__(self, x, y, data_type):
        """Initialize SVHNDataset."""
        self.data_type = data_type
        self.x = x
        self.y = y

    def __getitem__(self, index: int):
        """Return an item by the index."""
        return self.x[index], self.y[index]

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.x)


class SVHNShardDescriptor(ShardDescriptor):
    """SVHN Shard descriptor class."""

    def __init__(
            self,
            **kwargs
    ):
        """Initialize SVHNShardDescriptor."""
        (x_train, y_train), (x_test, y_test) = self.download_data()
        self.data_by_type = {
            'train': (x_train, y_train),
            'val': (x_test, y_test)
        }

    def get_shard_dataset_types(self) -> List[str]:
        """Get available shard dataset types."""
        return list(self.data_by_type)

    def get_dataset(self, dataset_type='train'):
        """Return a shard dataset by type."""
        if dataset_type not in self.data_by_type:
            raise Exception(f'Wrong dataset type: {dataset_type}')
        return SVHNShardDataset(
            *self.data_by_type[dataset_type],
            data_type=dataset_type
        )

    @property
    def sample_shape(self):
        """Return the sample shape info."""
        return ['28', '28', '1']

    @property
    def target_shape(self):
        """Return the target shape info."""
        return ['28', '28', '1']
               
    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return (f'SVHN dataset, single shard')

    def download_data(self):
        """Download prepared dataset."""
        #svhn_url = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
        #svhn_test_url = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'
        #response = requests.get(svhn_url)
        #response_test = requests.get(svhn_test_url)
        
        #train_data = sio.loadmat('train_32x32.mat')
        #test_data  = sio.loadmat('test_32x32.mat')
        #train_images = train_data['X'].transpose([3,0,1,2])   # from HWCN to NCHW
        #test_images = test_data['X'].transpose([3,0,1,2])   # from HWCN to NCHW
        #x_train, y_train = train_images, train_data['y']
        #x_test,  y_test  = test_images,  test_data['y'] 
        
        data_transform = transforms.Compose([
          transforms.Resize((32,32)),
          transforms.Grayscale(num_output_channels=1),
          transforms.ToTensor(),
        ])
        base_dir = 'http://ufldl.stanford.edu/housenumbers/'
        train_set = torchvision.datasets.SVHN(root=base_dir, split='train', download=True, transform=data_transform, target_transform=None)
        test_set = torchvision.datasets.SVHN(root=base_dir, split='test', download=True, transform=data_transform)
        train_data = train_set.data
        y_train = train_set.labels
        test_data = test_set.data
        y_test = test_set.labels
        x_train = train_data.transpose([0,3,2,1])
        x_test = test_data.transpose([0,3,2,1])
        x_train = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in x_train])
        x_test = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in x_test])
        print(x_train.shape)
        print(x_test.shape)
        print(y_train.shape)
        print(y_test.shape)

        print('SVHN data was loaded!')
        return (x_train, y_train), (x_test, y_test)