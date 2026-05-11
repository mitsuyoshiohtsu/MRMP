from __future__ import print_function

import os
import os.path
import subprocess
import sys
import zipfile

import numpy as np
from PIL import Image
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .utils import noisify


class UCMERCED(data.Dataset):
    base_folder = 'UCMerced_LandUse'
    url = "https://www.kaggle.com/api/v1/datasets/download/abdulhasibuddin/uc-merced-land-use-dataset"
    filename = "uc-merced-land-use-dataset.zip"

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False,
                 noise_type=None, noise_rate=0.2, random_state=0):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.dataset = 'ucmerced'
        self.noise_type = noise_type
        self.nb_classes = 21

        if download:
            self.download()

        classname = os.listdir('./data/UCMerced_LandUse/Images')
        img_paths = []
        classes_img = []
        for cls in classname:
            for sub in os.listdir(f'./data/UCMerced_LandUse/Images/{cls}'):
                img_paths.append(f'./data/UCMerced_LandUse/Images/{cls}/{sub}')
                classes_img.append(cls)

        labelencode = LabelEncoder()
        labels = labelencode.fit_transform(classes_img)

        train_data, test_data, train_labels, test_labels = train_test_split(
            img_paths, labels, test_size=0.2, random_state=42
        )

        if self.train:
            self.train_data = train_data
            self.train_labels = train_labels

            if noise_type in ["pairflip", "symmetric", "rns"]:
                self.train_labels = np.asarray(
                    [[self.train_labels[i]] for i in range(len(self.train_labels))]
                )
                self.train_noisy_labels, self.actual_noise_rate = noisify(
                    dataset=self.dataset, train_labels=self.train_labels,
                    noise_type=noise_type, noise_rate=noise_rate,
                    random_state=random_state, nb_classes=self.nb_classes,
                )
                self.train_noisy_labels = [i[0] for i in self.train_noisy_labels]
                _train_labels = [i[0] for i in self.train_labels]
                self.noise_or_not = (
                    np.transpose(self.train_noisy_labels) == np.transpose(_train_labels)
                )
            else:
                self.train_labels = np.asarray(
                    [[self.train_labels[i]] for i in range(len(self.train_labels))]
                )
                self.train_noisy_labels = self.train_labels.copy()
                self.actual_noise_rate = (
                    self.train_noisy_labels != self.train_labels
                ).mean()
                self.train_noisy_labels = [i for i in self.train_noisy_labels]
                _train_labels = [i[0] for i in self.train_labels]
                self.noise_or_not = (
                    np.transpose(self.train_noisy_labels) == np.transpose(_train_labels)
                )

            self.train_labels = np.array(self.train_labels)
            num_classes = np.max(self.train_labels) + 1
            self.train_labels = np.eye(num_classes)[self.train_labels]
            if noise_type != "rns":
                self.train_noisy_labels = np.eye(self.nb_classes)[self.train_noisy_labels]
            self.train_noisy_labels_s = self.train_noisy_labels.copy()
        else:
            self.test_data = test_data
            self.test_labels = test_labels
            self.test_labels = np.eye(self.nb_classes)[self.test_labels]

    def __getitem__(self, index):
        if self.train:
            if self.noise_type is not None:
                img, target, src = (
                    self.train_data[index],
                    self.train_noisy_labels[index],
                    self.train_noisy_labels_s[index],
                )
            else:
                img, target = self.train_data[index], self.train_labels[index]
                src = target
        else:
            img, target = self.test_data[index], self.test_labels[index]
            src = target

        img = Image.open(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, src, index

    def __len__(self):
        return len(self.train_data) if self.train else len(self.test_data)

    def download(self):
        curl_command = [
            "curl", "-L", "-o", "./data/uc-merced-land-use-dataset.zip",
            "https://www.kaggle.com/api/v1/datasets/download/abdulhasibuddin/uc-merced-land-use-dataset",
        ]
        subprocess.run(curl_command, capture_output=True, text=True)

        zip_path = './data/uc-merced-land-use-dataset.zip'
        extract_path = './data/'
        os.makedirs(extract_path, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

    def __repr__(self):
        fmt_str  = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format('train' if self.train else 'test')
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp))
        )
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp))
        )
        return fmt_str