import torch
import torch.nn.functional as F
# FIX #13: removed unused imports — random, torchvision.transforms.functional, GaussianMixture


def meta_update(meta_iter, model, optimizer, alpha):
    model.eval()

    # Sample one meta batch
    meta_images, _, meta_labels, _ = next(meta_iter)
    meta_images = meta_images.cuda()
    meta_labels = meta_labels.cuda()

    # Forward pass on meta (validation) split
    optimizer.zero_grad()
    val_logits = model(meta_images)
    val_loss = F.cross_entropy(val_logits, meta_labels)
    val_loss.backward()

    lr = optimizer.param_groups[0]["lr"]

    v = [lr * p.grad.clone() for p in model.parameters()]

    optimizer.zero_grad()

    # Normalised step size: epsilon = alpha / ||v||
    norm_v = torch.sqrt(sum((vi ** 2).sum() for vi in v))
    # FIX: guard against zero norm to avoid inf epsilon
    epsilon = alpha / norm_v.clamp(min=1e-8)

    return v, epsilon, val_loss.item()
