from __future__ import print_function

import os
import os.path
import sys

import numpy as np
from PIL import Image

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch
import torch.utils.data as data
from torchvision.datasets import CIFAR10, CIFAR100

from .utils import check_integrity, download_url, noisify


class CIFAR10(data.Dataset):
    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"

    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]
    test_list = [["test_batch", "40351d587109b95175f43aff81a1287e"]]

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, noise_type=None, noise_rate=0.2, random_state=0):

        self.root = os.path.expanduser(root)
        self.transform, self.target_transform = transform, target_transform
        self.train = train
        self.dataset = "cifar10"
        self.noise_type = noise_type
        self.nb_classes = 10

        if download:
            self.download()
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. Use download=True.")

        if self.train:
            self._load_train_data()
            self._inject_noise(noise_type, noise_rate, random_state)
            self._one_hot_train_labels()
        else:
            self._load_test_data()
            self._one_hot_test_labels()

    def _load_file(self, file):
        with open(file, "rb") as fo:
            return pickle.load(fo, encoding="latin1")

    def _load_train_data(self):
        data_list = []
        labels_list = []
    
        # loop all train files
        for file_name, checksum in self.train_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
    
            entry = self._load_file(file_path)
            data = entry["data"]
            labels = entry["labels"] 
    
            data_list.append(data)
            labels_list.extend(labels)
    
        # concat all data
        data = np.vstack(data_list)  # (50000, 3072)
        labels = np.array(labels_list)
    
        num = data.shape[0]
        self.train_data = data.reshape((num, 3, 32, 32)).transpose(0, 2, 3, 1)
        self.train_labels = labels.reshape(-1, 1)

    def _load_test_data(self):
        fname, _ = self.test_list[0]
        entry = self._load_file(os.path.join(self.root, self.base_folder, fname))
        data = entry["data"]
        labels = entry["labels"] 
        num = data.shape[0]  # automatically detect number of images
        self.test_data = data.reshape((num, 3, 32, 32)).transpose(0, 2, 3, 1)
        self.test_labels = np.array(labels)

    def _inject_noise(self, noise_type, noise_rate, random_state):
        if noise_type in ["pairflip", "symmetric"]:
            noisy, actual = noisify(
                dataset=self.dataset,
                train_labels=self.train_labels,
                noise_type=noise_type,
                noise_rate=noise_rate,
                random_state=random_state,
                nb_classes=self.nb_classes,
            )
            self.train_noisy_labels = noisy[:, 0]
            self.actual_noise_rate = actual
        else:
            with torch.serialization.safe_globals([np.core.multiarray._reconstruct]):
                cifar10N = torch.load("../CIFAR-10_human.pt", weights_only=False)
            self.train_noisy_labels = np.array(cifar10N[noise_type])
            self.actual_noise_rate = (self.train_noisy_labels != self.train_labels[:, 0]).mean()

        self.noise_or_not = (self.train_noisy_labels == self.train_labels[:, 0])
        self.train_noisy_labels_s = self.train_noisy_labels.copy()

    def _one_hot_train_labels(self):
        n = self.nb_classes
        self.train_labels        = np.eye(n)[self.train_labels[:, 0]]
        self.train_noisy_labels  = np.eye(n)[self.train_noisy_labels]
        self.train_noisy_labels_s = np.eye(n)[self.train_noisy_labels_s]

    def _one_hot_test_labels(self):
        n = self.nb_classes
        self.test_labels = np.eye(n)[self.test_labels]
        
    def __getitem__(self, index):
        if self.train:
            img = self.train_data[index]
            noisy = self.train_noisy_labels[index]
            src = self.train_noisy_labels_s[index]
        else:
            img = self.test_data[index]
            noisy = src = self.test_labels[index]

        img = Image.fromarray(img)
        if self.transform: img = self.transform(img)
        if self.target_transform: noisy = self.target_transform(noisy)
        return img, noisy, src, index

    def __len__(self): 
        return len(self.train_data) if self.train else len(self.test_data)


    def _check_integrity(self):
        for fname, md5 in self.train_list + self.test_list:
            fpath = os.path.join(self.root, self.base_folder, fname)
            if not check_integrity(fpath, md5): return False
        return True

    def download(self):
        if self._check_integrity():
            print("Files already downloaded and verified"); return
        download_url(self.url, self.root, self.filename, self.tgz_md5)
        import tarfile
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(self.root)

    def __repr__(self):
        split = "train" if self.train else "test"
        t = lambda x: x.__repr__().replace("\n", "\n" + " " * 25)
        return (f"Dataset CIFAR10\n"
                f"    Datapoints: {self.__len__()}\n"
                f"    Split: {split}\n"
                f"    Root: {self.root}\n"
                f"    Transform: {t(self.transform)}\n"
                f"    Target Transform: {t(self.target_transform)}")


class CIFAR100(data.Dataset):
    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"

    train_list = [["train", "16019d7e3df5f24257cddd939b257f8d"]]
    test_list  = [["test",  "f0ef6b0ae62326f3e7ffdfab6717acfc"]]

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, noise_type=None, noise_rate=0.2, random_state=0):

        self.root = os.path.expanduser(root)
        self.transform, self.target_transform = transform, target_transform
        self.train = train
        self.dataset = "cifar100"
        self.noise_type = noise_type
        self.nb_classes = 100

        if download: self.download()
        if not self._check_integrity():
            raise RuntimeError("Dataset missing or corrupted. Use download=True.")

        if self.train:
            self._load_train_data()
            self._inject_noise(noise_type, noise_rate, random_state)
            self._one_hot_train_labels()
        else:
            self._load_test_data()
            self._one_hot_test_labels()

    def _load_file(self, file):
        with open(file, "rb") as fo:
            return pickle.load(fo, encoding="latin1")

    def _load_train_data(self):
        entry = self._load_file(os.path.join(self.root, self.base_folder, self.train_list[0][0]))
        
        data = entry["data"]
        labels = entry["fine_labels"]  # CIFAR-100

        print(entry["data"].shape)
    
        num = data.shape[0]  # automatically detect number of images
        self.train_data = data.reshape((num, 3, 32, 32)).transpose(0, 2, 3, 1)
        self.train_labels = np.array(labels).reshape(-1, 1)
    
    def _load_test_data(self):
        entry = self._load_file(os.path.join(self.root, self.base_folder, self.test_list[0][0]))
    
        data = entry["data"]
        labels = entry["fine_labels"]  # CIFAR-100
    
        num = data.shape[0]  # automatically detect number of images
        self.test_data = data.reshape((num, 3, 32, 32)).transpose(0, 2, 3, 1)
        self.test_labels = np.array(labels)

    def _inject_noise(self, noise_type, noise_rate, random_state):
        if noise_type != "noisy_label":
            noisy, actual = noisify(
                dataset=self.dataset,
                train_labels=self.train_labels,
                noise_type=noise_type,
                noise_rate=noise_rate,
                random_state=random_state,
                nb_classes=self.nb_classes,
            )
            self.train_noisy_labels = noisy[:, 0]
            self.actual_noise_rate = actual
        else:
            with torch.serialization.safe_globals([np.core.multiarray._reconstruct]):
                cifar100N = torch.load("../CIFAR-100_human.pt", weights_only=False)
            self.train_noisy_labels = np.array(cifar100N[noise_type])
            self.actual_noise_rate = (self.train_noisy_labels != self.train_labels[:, 0]).mean()

        self.train_noisy_labels_s = self.train_noisy_labels.copy()
        self.noise_or_not = (self.train_noisy_labels == self.train_labels[:, 0])

    def _one_hot_train_labels(self):
        n = self.nb_classes
        self.train_labels        = np.eye(n)[self.train_labels[:, 0]]
        self.train_noisy_labels  = np.eye(n)[self.train_noisy_labels]
        self.train_noisy_labels_s = np.eye(n)[self.train_noisy_labels_s]

    def _one_hot_test_labels(self):
        n = self.nb_classes
        self.test_labels = np.eye(n)[self.test_labels]

    def __getitem__(self, index):
        if self.train:
            img = self.train_data[index]
            noisy = self.train_noisy_labels[index]
            src = self.train_noisy_labels_s[index]
        else:
            img = self.test_data[index]
            noisy = src = self.test_labels[index]

        img = Image.fromarray(img)
        if self.transform: img = self.transform(img)
        if self.target_transform:
            noisy = self.target_transform(noisy)
            src   = self.target_transform(src)
        return img, noisy, src, index

    def __len__(self): 
        return len(self.train_data) if self.train else len(self.test_data)

    def _check_integrity(self):
        for fname, md5 in self.train_list + self.test_list:
            fpath = os.path.join(self.root, self.base_folder, fname)
            if not check_integrity(fpath, md5): return False
        return True

    def download(self):
        if self._check_integrity():
            print("Files already downloaded and verified"); return
        download_url(self.url, self.root, self.filename, self.tgz_md5)
        import tarfile
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(self.root)

    def __repr__(self):
        sp = "train" if self.train else "test"
        tf  = self.transform.__repr__().replace("\n", "\n" + " " * 25)
        ttf = self.target_transform.__repr__().replace("\n", "\n" + " " * 25)
        return (f"Dataset CIFAR100\n"
                f"    Datapoints: {self.__len__()}\n"
                f"    Split: {sp}\n"
                f"    Root: {self.root}\n"
                f"    Transform: {tf}\n"
                f"    Target Transform: {ttf}")