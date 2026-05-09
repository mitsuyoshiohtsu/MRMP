import random

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from sklearn.mixture import GaussianMixture


def meta_update(meta_iter, model, optimizer, alpha):
    model.eval()
    meta_images, _, meta_labels, _ = next(meta_iter)
    meta_images = meta_images.cuda()
    meta_labels = meta_labels.cuda()

    optimizer.zero_grad()
    val_logits = model(meta_images)
    val_loss = F.cross_entropy(val_logits, meta_labels)
    val_loss.backward()

    lr = optimizer.param_groups[0]["lr"]
    v = [lr * p.grad for p in model.parameters()]
    optimizer.zero_grad()

    norm_v = torch.sqrt(sum((vi ** 2).sum() for vi in v))
    epsilon = alpha / norm_v
    return v, epsilon, val_loss.item()