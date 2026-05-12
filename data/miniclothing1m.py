from __future__ import print_function

import os
import numpy as np
from PIL import Image

import torch.utils.data as data


class MiniClothing1M(data.Dataset):

    NB_CLASSES = 14
    SPLIT_TO_FILE = {
        "train": "clean_noisy_labels_train.txt",
        "val":   "clean_noisy_labels_val.txt",
        "test":  "clean_noisy_labels_test.txt",
        "total": "clean_noisy_labels_total.txt",
    }

    def __init__(self, root, split="train", transform=None, target_transform=None):
        """
        Args:
            root:             Path to mini_clothing1M root (contains label .txt files
                              and images/ subdirectory).
            split:            One of 'train', 'val', 'test', 'total'.
            transform:        Transform applied to the PIL image.
            target_transform: Transform applied to the noisy label.
        """
        assert split in self.SPLIT_TO_FILE, \
            f"split must be one of {list(self.SPLIT_TO_FILE.keys())}, got '{split}'"

        self.root             = os.path.expanduser(root)
        self.split            = split
        self.transform        = transform
        self.target_transform = target_transform

        label_file = os.path.join(self.root, self.SPLIT_TO_FILE[split])
        self.img_paths, self.clean_labels, self.noisy_labels = self._load_labels(label_file)

        # Boolean mask: True where noisy == clean
        self.noise_or_not = (
            np.array(self.clean_labels) == np.array(self.noisy_labels)
        )

        # One-hot encoded versions
        eye = np.eye(self.NB_CLASSES, dtype=np.float32)
        self.train_labels  = eye[self.clean_labels]
        self.train_noisy_labels  = eye[self.noisy_labels]
        self.train_noisy_labels_s = self.train_noisy_labels.copy()

    # ------------------------------------------------------------------
    def _load_labels(self, label_file):
        """Parse label file → (img_paths, clean_labels, noisy_labels).

        Label file format (one per line):
            images/3/01/filename.jpg  <noisy_label>  <clean_label>
        """
        img_paths, clean_labels, noisy_labels = [], [], []
        with open(label_file) as f:
            for lineno, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) != 3:
                    print(f"  [!] Skipping malformed line {lineno} in {label_file}")
                    continue
                rel_path, noisy, clean = parts
                full_path = os.path.join(self.root, rel_path)
                img_paths.append(full_path)
                clean_labels.append(int(clean))
                noisy_labels.append(int(noisy))
        return img_paths, clean_labels, noisy_labels

    # ------------------------------------------------------------------
    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert("RGB")

        noisy_label = self.train_noisy_labels[index]
        src_label = self.train_noisy_labels_s[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            noisy_label = self.target_transform(noisy_label)

        return img, noisy_label, src_label, index

    def __len__(self):
        return len(self.img_paths)

    def __repr__(self):
        noise_rate = (~self.noise_or_not).mean() * 100
        return (
            f"Dataset {self.__class__.__name__}\n"
            f"    Split:            {self.split}\n"
            f"    Samples:          {len(self)}\n"
            f"    Root:             {self.root}\n"
            f"    Noise rate:       {noise_rate:.1f}%\n"
            f"    Transform:        {self.transform}\n"
            f"    Target transform: {self.target_transform}\n"
        )