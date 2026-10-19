import torch
import torch.nn.functional as F
# FIX #13: removed unused imports — random, torchvision.transforms.functional, GaussianMixture


def meta_update(meta_iter, model, optimizer, alpha):
    """
    Compute the finite-difference meta-gradient direction v and step size epsilon.

    Returns:
        v       : list of per-parameter gradient vectors (same shape as model params)
        epsilon : finite-difference step size  (alpha / ||v||)
        val_loss: scalar validation loss used to compute v (float)

    Notes:
        - FIX #14: model.eval() is called here but the caller is responsible for
          restoring model.train() afterwards if needed. The training-mode flag is
          NOT restored inside this function to keep it stateless, and the caller
          (meta_train in MRMP.py) already manages training modes explicitly.
        - v tensors are detached from the graph (p.grad.clone()) so they are safe
          to use as perturbation vectors without retaining graph memory.
        - epsilon is guarded against division by zero (norm_v clamp).
    """
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

    # v = lr * gradient  (finite-difference perturbation direction)
    # FIX: clone gradients before zeroing so v is not wiped
    v = [lr * p.grad.clone() for p in model.parameters()]

    optimizer.zero_grad()

    # Normalised step size: epsilon = alpha / ||v||
    norm_v = torch.sqrt(sum((vi ** 2).sum() for vi in v))
    # FIX: guard against zero norm to avoid inf epsilon
    epsilon = alpha / norm_v.clamp(min=1e-8)

    return v, epsilon, val_loss.item()
