from __future__ import print_function

import argparse
import datetime
import json
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from ema_pytorch import EMA
from scipy.optimize import curve_fit  
from sklearn.model_selection import KFold
from tqdm import tqdm
import timm

from data.cifar import CIFAR10, CIFAR100
from meta_modules import meta_update
from model import NoiseDetector

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser("PyTorch CIFAR Training")
parser.add_argument("--dataset", type=str, default="cifar100", help="dataset")
parser.add_argument("--noise_rate", type=float, default=0.0, help="noise rate")
parser.add_argument("--noise_type", type=str, default="symmetric", help="noise type")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--n_epoch", type=int, default=40, help="epochs")
parser.add_argument("--num_iter_per_epoch", type=int, default=1000, help="iters/epoch")
parser.add_argument("--n_round", type=int, default=10, help="rounds")
parser.add_argument("--start_round", type=int, default=1, help="start round")
parser.add_argument("--meta_iter", type=int, default=1000, help="meta iters")
parser.add_argument("--beta", type=float, default=0.9, help="EMA beta")
parser.add_argument("--T", type=float, default=0.7, help="temp")
parser.add_argument("--ES_step", type=int, default=5, help="early stop step")
parser.add_argument("--result_dir", type=str, default="results/", help="save dir")
parser.add_argument("--load", type=str, default=None, help="load purified dataset")
parser.add_argument("--gpuid", type=int, default=0, help="GPU id")
parser.add_argument("--num_workers", type=int, default=4, help="workers")
parser.add_argument("--print_freq", type=int, default=50, help="print freq")
parser.add_argument("--seed", type=int, default=123, help="seed")
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)

# Seed
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

model_str = (
    f"{args.dataset}_MRML_{args.noise_type}_{args.noise_rate}"
    f"_beta{args.beta}_tau{args.T}"
)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
if args.dataset == "cifar10":
    num_classes = 10
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    try:
        train_dataset = torch.load(args.load, weights_only=False)
        train_dataset.transform = transform_train
        print("Loaded existing dataset.")
    except Exception:
        train_dataset = CIFAR10(
            root="./data/", download=True, train=True,
            transform=transform_train,
            noise_type=args.noise_type, noise_rate=args.noise_rate,
        )
    test_dataset = CIFAR10(
        root="./data/", download=True, train=False,
        transform=transform_test,
        noise_type=args.noise_type, noise_rate=args.noise_rate,
    )

elif args.dataset == "cifar100":
    num_classes = 100
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])
    try:
        train_dataset = torch.load(args.load, weights_only=False)  
        train_dataset.transform = transform_train
        print("Loaded existing dataset.")
    except Exception:
        train_dataset = CIFAR100(
            root="./data/", download=True, train=True,
            transform=transform_train,
            noise_type=args.noise_type, noise_rate=args.noise_rate,
        )
    test_dataset = CIFAR100(
        root="./data/", download=True, train=False,
        transform=transform_test,
        noise_type=args.noise_type, noise_rate=args.noise_rate,
    )

else:
    raise ValueError(f"Unknown dataset: {args.dataset}")

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
save_dir = os.path.join(args.result_dir, args.dataset, "MRML")
os.makedirs(save_dir, exist_ok=True)

txtfile = os.path.join(save_dir, model_str + ".txt")
nowTime = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
if os.path.exists(txtfile):
    os.rename(txtfile, txtfile + f".bak-{nowTime}")

jsonfile = os.path.join(save_dir, model_str + ".json")
if os.path.exists(jsonfile) and args.load is None:
    os.rename(jsonfile, jsonfile + f".bak-{nowTime}")

# ---------------------------------------------------------------------------
# Curve-fit helpers (defined at module level, not inside the loop)
# ---------------------------------------------------------------------------
def exp_func_fixed_c(x, a, b):
    return np.log(a) - b * x

def pred_round(a, b, c):
    return (1.0 / b) * np.log(a / c)

# ---------------------------------------------------------------------------
# Accuracy helper
# ---------------------------------------------------------------------------
def accuracy(logit, target, topk=(1,)):
    """Computes precision@k for the specified values of k."""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(train_loader, epoch, model, optimizer):
    model.train()
    train_total = 0
    train_correct = 0

    for i, (images, labels, s_labels, indexes) in tqdm(
        enumerate(train_loader),
        total=args.num_iter_per_epoch,
        desc="Training",
        unit="batch",
    ):
        if i >= args.num_iter_per_epoch:  # FIX: was '>' allowing one extra iter
            break

        images = images.cuda()
        labels = labels.cuda()

        logits = model(images)
        prec1, _ = accuracy(logits, torch.max(labels.data, 1)[1], topk=(1, 5))
        train_total += 1
        train_correct += prec1

        loss = F.cross_entropy(logits, labels)

        # Prior entropy penalty (encourages uniform class predictions)
        prior = torch.ones(num_classes, device=images.device) / num_classes
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))
        loss = loss + penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_acc = float(train_correct) / float(train_total)
    return train_acc, loss.item()

# ---------------------------------------------------------------------------
# Meta-training
# ---------------------------------------------------------------------------
def meta_train(train_loader, meta_loader, max_iter, model1, optimizer, model2, nd, noise_optimizer):
    print("Meta-Training...")
    meta_count = 0
    min_loss = 1e9

    train_iter = iter(train_loader)
    meta_iter = iter(meta_loader)

    model1.eval()
    nd.train()

    for i in tqdm(range(max_iter), desc="Meta-Training"):  
        # Sample from train split
        try:
            images, labels, s_labels, indexes = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, labels, s_labels, indexes = next(train_iter)

        images = images.cuda()
        labels = labels.cuda()

        # EMA teacher pseudo-labels (no grad needed from teacher)
        with torch.no_grad():
            logits2 = model2(images)
            c_labels = F.softmax(logits2, dim=1)

        # Meta-gradient: finite-difference approximation
        try:
            v, epsilon, val_loss = meta_update(meta_iter, model1, optimizer, alpha=1.0)
        except StopIteration:
            meta_iter = iter(meta_loader)
            v, epsilon, val_loss = meta_update(meta_iter, model1, optimizer, alpha=1.0)

        noise_optimizer.zero_grad()

        meta_loss = torch.zeros(1, device=images.device)

        for sign in [+1, -1]:
            # Perturb model1 parameters along v
            with torch.no_grad():
                for p, v_p in zip(model1.parameters(), v):
                    p.add_(v_p, alpha=sign * epsilon)

            logits = model1(images)
            next_labels = nd(c_labels, labels)
            loss = F.cross_entropy(logits, next_labels)
            ((sign * loss) / (2 * epsilon)).backward()             # ← sign-weighted backward
            meta_loss = meta_loss + (sign * loss).detach() / (2 * epsilon)

            # Restore model1 parameters
            with torch.no_grad():
                for p, v_p in zip(model1.parameters(), v):
                    p.sub_(v_p, alpha=sign * epsilon)

        torch.nn.utils.clip_grad_norm_(nd.parameters(), max_norm=5.0)
        noise_optimizer.step()

        # Early stopping for meta-training
        meta_loss_val = meta_loss.item()  
        if meta_loss_val > min_loss:
            meta_count += 1
        else:
            meta_count = 0
            min_loss = meta_loss_val

        if meta_count > 100:
            break

    print("Meta-Training complete.")

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(loader, model):
    model.eval()
    correct = 0
    total = 0
    last_loss = 0.0

    with torch.no_grad():
        for images, _, labels, _ in loader:
            images = images.cuda()
            logits = model(images)
            last_loss = F.cross_entropy(logits.cpu(), labels).item()
            _, pred = torch.max(F.softmax(logits, dim=1).data, 1)
            _, labels = torch.max(labels.data, 1)
            total += labels.size(0)
            correct += (pred.cpu() == labels).sum().item()

    acc = 100.0 * correct / total
    return acc, last_loss

# ---------------------------------------------------------------------------
# Dataset purification
# ---------------------------------------------------------------------------
def purify_dataset(model, ema, nd, dataset, s_dataset, json_f):
    model.eval()
    ema.eval()
    nd.eval()

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    device = next(model.parameters()).device

    n2c_flip = n2n_flip = c2n_flip = 0
    mean_kl = 0.0
    kl_batches = 0

    with torch.no_grad():
        for images, labels, s_labels, idxs in tqdm(loader, desc="Purifying"):
            images = images.to(device)
            labels = labels.to(device)

            # EMA pseudo-labels
            logits2 = ema(images)
            c_labels = F.softmax(logits2, dim=1)

            # Purified soft labels from NoiseDetector
            pur = nd(c_labels, labels)  # (B, C)

            # KL divergence between purified and current noisy labels
            pur_clamped = pur.clamp(min=1e-8)
            kl = F.kl_div(pur_clamped.log(), labels.clamp(min=1e-8), reduction="batchmean")
            mean_kl += kl.item()
            kl_batches += 1

            # Update dataset labels in-place
            pur_np = pur.cpu().numpy()
            for j, subset_idx in enumerate(idxs.numpy()):
                old_label = np.argmax(s_dataset.train_noisy_labels_s[subset_idx])
                new_label = np.argmax(pur_np[j])
                clean_label = np.argmax(s_dataset.train_labels[subset_idx])

                s_dataset.train_noisy_labels[subset_idx] = pur_np[j]
                s_dataset.noise_or_not[subset_idx] = (new_label == clean_label)

                if old_label != new_label:
                    if new_label == clean_label:
                        n2c_flip += 1
                    elif old_label == clean_label:
                        c2n_flip += 1
                    else:
                        n2n_flip += 1

    n_total = len(s_dataset.noise_or_not)
    n_true = int(np.sum(s_dataset.noise_or_not))
    data_acc = n_true / n_total
    avg_kl = mean_kl / kl_batches

    json_f["data_acc"].append(data_acc)
    json_f["n2c_flip"].append(n2c_flip)
    json_f["n2n_flip"].append(n2n_flip)
    json_f["c2n_flip"].append(c2n_flip)
    json_f["mean_kl"].append(avg_kl)

    print(
        f"Purified ({n_total} samples) | acc={data_acc:.4f} | "
        f"n2c={n2c_flip} n2n={n2n_flip} c2n={c2n_flip} | mean_kl={avg_kl:.6f}"
    )
    return json_f

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
with open(txtfile, "a") as f:
    f.write("epoch train_acc train_loss meta_acc meta_acc_ema meta_loss meta_loss_ema "
            "test_acc test_acc_ema test_loss test_loss_ema\n")

torch.cuda.reset_peak_memory_stats()

if args.load is not None:
    with open(jsonfile, "r") as f:
        json_f = json.load(f)
else:
    json_f = {
        "data_acc": [],
        "n2c_flip": [],
        "n2n_flip": [],
        "c2n_flip": [],
        "mean_kl": [],
        "pred_opt_r": [],
    }

kf = KFold(n_splits=args.n_round, shuffle=True, random_state=42)

for r in range(args.start_round, args.n_round + 1):

    # Select the r-th fold as meta split
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
        if fold == r - 1:
            base_dataset = torch.utils.data.Subset(train_dataset, train_idx)
            meta_dataset = torch.utils.data.Subset(train_dataset, val_idx)
            break

    print(f"\n{'='*60}")
    print(f"Round {r}/{args.n_round}  |  base={len(base_dataset)}  meta={len(meta_dataset)}")
    print(f"{'='*60}")

    train_loader = torch.utils.data.DataLoader(
        base_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, drop_last=True, shuffle=True,
    )
    meta_loader = torch.utils.data.DataLoader(
        meta_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, drop_last=True, shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, drop_last=False, shuffle=False,
    )

    cudnn.benchmark = True

    # Build model
    if args.dataset == "cifar100" and args.noise_rate == 0.7:
        cnn = timm.create_model("convnext_base", pretrained=True, num_classes=num_classes).cuda()
    else:
        cnn = timm.create_model("convnext_small", pretrained=True, num_classes=num_classes).cuda()

    cnn_ema = EMA(cnn, beta=args.beta, update_after_step=0, update_every=1)
    cnn_ema.update()

    optimizer = optim.SGD(cnn.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    matching = (
        train_dataset.train_noisy_labels.argmax(1)
        == train_dataset.train_labels.argmax(1)
    )
    print(f"Current label accuracy: {matching.mean():.4f}")

    # ------------------------------------------------------------------
    # Pre-training with early stopping
    # ------------------------------------------------------------------
    val_acc_max = 0.0
    count = 0

    for epoch in range(1, args.n_epoch + 1):
        cnn.train()
        train_acc, train_loss = train(train_loader, epoch, cnn, optimizer)
        cnn_ema.update()

        meta_acc, meta_loss         = evaluate(meta_loader, cnn)
        meta_acc_ema, meta_loss_ema = evaluate(meta_loader, cnn_ema)
        test_acc, test_loss         = evaluate(test_loader, cnn)
        test_acc_ema, test_loss_ema = evaluate(test_loader, cnn_ema)

        print(
            f"Epoch [{epoch}/{args.n_epoch}] | "
            f"Test: {test_acc:.2f}% (EMA {test_acc_ema:.2f}%) | "
            f"Loss: {test_loss:.4f} (EMA {test_loss_ema:.4f})"
        )

        with open(txtfile, "a") as f:
            f.write(
                f"{epoch} {train_acc:.4f} {train_loss:.4f} "
                f"{meta_acc:.4f} {meta_acc_ema:.4f} {meta_loss:.4f} {meta_loss_ema:.4f} "
                f"{test_acc:.4f} {test_acc_ema:.4f} {test_loss:.4f} {test_loss_ema:.4f}\n"
            )

        # Early stopping
        if meta_acc >= val_acc_max:
            val_acc_max = meta_acc
            count = 0
        else:
            count += 1
            if count > args.ES_step:
                print(f"Early stopping at epoch {epoch}.")
                break

    # ------------------------------------------------------------------
    # NoiseDetector + Meta-training
    # ------------------------------------------------------------------
    hidden_dim = 16 if args.dataset == "cifar10" else 32
    nd = NoiseDetector(input_dim=num_classes, hidden_dim=hidden_dim, temperature=args.T).cuda()
    noise_opt = optim.SGD(nd.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

    cnn.train()
    cnn_ema.eval()
    nd.train()

    meta_train(train_loader, meta_loader, args.meta_iter, cnn, optimizer, cnn_ema, nd, noise_opt)

    # ------------------------------------------------------------------
    # Purify dataset
    # ------------------------------------------------------------------
    json_f = purify_dataset(cnn, cnn_ema, nd, base_dataset, train_dataset, json_f)
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")

    # ------------------------------------------------------------------
    # Convergence check via exponential curve fit on KL history
    # ------------------------------------------------------------------
    mean_kl_now = json_f["mean_kl"][-1]
    kl_history = np.array(json_f["mean_kl"])

    if mean_kl_now > kl_history.min() or r == args.n_round:
        if kl_history.argmin() >= 2:  # need at least 3 points to fit
            x_all = np.arange(1, len(kl_history) + 1, dtype=float)
            c_fixed = kl_history.min()
            x_fit = x_all[:kl_history.argmin()]
            y_fit = kl_history[:kl_history.argmin()]

            # Guard against values too close to c_fixed (log of ~0)
            mask = (y_fit - c_fixed) > 1e-10
            if mask.sum() >= 2:
                try:
                    y_log = np.log(y_fit[mask] - c_fixed)
                    popt, _ = curve_fit(exp_func_fixed_c, x_fit[mask], y_log, p0=(0.02, 1.0))
                    a_fit, b_fit = popt
                    pred_r = pred_round(a_fit, b_fit, c_fixed)
                    json_f["pred_opt_r"].append(float(pred_r))
                    opt_r = int(pred_r) + 1
                    print(f"Predicted optimal round: {pred_r:.2f}  →  stopping at r>{opt_r}")
                    if r > opt_r:
                        print("Convergence reached. Stopping purification loop.")
                        # Save before breaking
                        os.makedirs("./data/purified", exist_ok=True)
                        torch.save(train_dataset, f"./data/purified/{model_str}_at_round{r}.pt")
                        with open(jsonfile, "w") as f:
                            json.dump(json_f, f, indent=2)
                        torch.cuda.empty_cache()
                        break
                except RuntimeError:
                    print("curve_fit did not converge; continuing to next round.")

    # ------------------------------------------------------------------
    # Save purified dataset and logs
    # ------------------------------------------------------------------
    os.makedirs("./data/purified", exist_ok=True)
    save_path = f"./data/purified/{model_str}_at_round{r}.pt"
    torch.save(train_dataset, save_path)
    print(f"Saved purified dataset → {save_path}")

    with open(jsonfile, "w") as f:
        json.dump(json_f, f, indent=2)

    torch.cuda.empty_cache()

print("\nAll rounds complete.")
