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
from tqdm import tqdm

from data.cifar import CIFAR10, CIFAR100
from meta_modules import *
from model import NoiseDetector

from sklearn.model_selection import KFold
import timm

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
parser.add_argument("--load", type=str, default=None, help="load")
parser.add_argument("--gpuid", type=int, default=0, help="GPU id")
parser.add_argument("--num_workers", type=int, default=4, help="workers")
parser.add_argument("--print_freq", type=int, default=50, help="print freq")
parser.add_argument("--seed", type=int, default=123, help="seed")
args = parser.parse_args()


torch.cuda.set_device(args.gpuid)

# Hyper Parameters
batch_size = args.batch_size
learning_rate = args.lr

# Seed
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

model_str = f"{args.dataset}_MRML_{args.noise_type}_{args.noise_rate}_beta{args.beta}_tau{args.T}"

if args.dataset == "cifar10":
    input_channel = 3
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
        print("loaded existing dataset")
    except:
        train_dataset = CIFAR10(
            root="./data/",
            download=True,
            train=True,
            transform=transform_train,
            noise_type=args.noise_type,
            noise_rate=args.noise_rate,
        )

    test_dataset = CIFAR10(
        root="./data/",
        download=True,
        train=False,
        transform=transform_test,
        noise_type=args.noise_type,
        noise_rate=args.noise_rate,
    )

if args.dataset == "cifar100":
    input_channel = 3
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
        train_dataset = torch.load(args.load)
        train_dataset.transform = transform_train
        print("loaded existing dataset")
    except:
        train_dataset = CIFAR100(
            root="./data/",
            download=True,
            train=True,
            transform=transform_train,
            noise_type=args.noise_type,
            noise_rate=args.noise_rate,
        )

    test_dataset = CIFAR100(
        root="./data/",
        download=True,
        train=False,
        transform=transform_test,
        noise_type=args.noise_type,
        noise_rate=args.noise_rate,
    )

noise_or_not = train_dataset.noise_or_not

save_dir = args.result_dir + "/" + args.dataset + "/MRML/"

if not os.path.exists(save_dir):
    os.system("mkdir -p %s" % save_dir)

txtfile = save_dir + "/" + model_str + ".txt"
nowTime = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
if os.path.exists(txtfile):
    os.system("mv %s %s" % (txtfile, txtfile + ".bak-%s" % nowTime))

jsonfile = save_dir + "/" + model_str + ".json"
nowTime = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
if os.path.exists(jsonfile):
    if args.load != None:
        pass
    else:
        os.system("mv %s %s" % (jsonfile, jsonfile + ".bak-%s" % nowTime))

def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
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

# Train the Model
def train(train_loader, epoch, model1, optimizer):
    train_total = 0
    train_correct = 0

    for i, (images, labels, s_labels, indexes) in tqdm(
        enumerate(train_loader),
        total=args.num_iter_per_epoch,
        desc="Training",
        unit="batch",
    ):
        indexes.cpu().numpy().transpose()
        if i > args.num_iter_per_epoch:
            break

        images, labels, s_labels = images.cuda(), labels.cuda(), s_labels.cuda()
        
        # Forward + Backward + Optimize
        logits1 = model1(images)
        prec1, _ = accuracy(logits1, torch.max(labels.data, 1)[1], topk=(1, 5))
        train_total += 1
        train_correct += prec1
        loss = F.cross_entropy(logits1, labels)
        num_classes = logits1.size()[1]
        prior = torch.ones(num_classes) / num_classes
        prior = prior.cuda()
        pred_mean = torch.softmax(logits1, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))
        loss += penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_acc = float(train_correct) / float(train_total)
    return train_acc, loss.item()


def meta_train(meta_loader, max_iter, model1, optimizer, model2, nd, noise_optimizer):
    print("Meta-Training...")
    meta_count, min_loss = 0, 1e9

    train_iter = iter(train_loader)
    model1.eval(); nd.train()
    
    meta_iter = iter(meta_loader)

    for i in tqdm(range(len(train_loader)), total=max_iter):
        if i > max_iter: break

        try:
            images, labels, s_labels, indexes = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, labels, s_labels, indexes = next(train_iter)

        images, labels, s_labels = images.cuda(), labels.cuda(), s_labels.cuda()

        with torch.no_grad():
            logits2 = model2(images).detach()
            c_labels = F.softmax(logits2, dim=1)

        try:
            v, epsilon, val_loss = meta_update(meta_iter, model1, optimizer, alpha=1.0)
        except StopIteration:
            meta_iter = iter(meta_loader)
            v, epsilon, val_loss = meta_update(meta_iter, model1, optimizer, alpha=1.0)

        noise_optimizer.zero_grad()
        meta_loss = torch.zeros(1, device=images.device)

        for sign in [+1, -1]:
            with torch.no_grad():
                for p, v_p in zip(model1.parameters(), v):
                    p.add_(v_p, alpha=sign * epsilon)

            logits = model1(images)
            next_labels = nd(c_labels, labels)
            loss = F.cross_entropy(logits.detach(), next_labels)
            meta_loss += (sign * loss) / (2 * epsilon)
            loss.backward()

            with torch.no_grad():
                for p, v_p in zip(model1.parameters(), v):
                    p.sub_(v_p, alpha=sign * epsilon)

        torch.nn.utils.clip_grad_norm_(nd.parameters(), max_norm=5.0)
        noise_optimizer.step()

        if meta_loss > min_loss: meta_count += 1
        else: meta_count, min_loss = 0, val_loss
        if meta_count > 100: break

    print("Meta-Training complete.")

# Evaluate the Model
def evaluate(test_loader, model):
    model.eval()
    correct, total, last_loss = 0, 0, 0
    for images, _, labels, _ in test_loader:
        images = images.cuda()
        logits = model(images)
        last_loss = F.cross_entropy(logits.cpu(), labels)
        _, pred = torch.max(F.softmax(logits, dim=1).data, 1)
        _, labels = torch.max(labels.data, 1)
        total += labels.size(0)
        correct += (pred.cpu() == labels).sum()
    acc = 100 * float(correct) / float(total)
    return acc, last_loss.item()

# Purify the Dataset
def purify_dataset(model, ema, nd, Dataset, S_Dataset, json_f):
    model.eval(); ema.eval(); nd.eval()
    Loader = torch.utils.data.DataLoader(Dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    device = next(model.parameters()).device

    n2c_flip = n2n_flip = c2n_flip = 0
    mean_kl = 0.0; kl_batches = 0

    with torch.no_grad():
        for images, labels, s_labels, idxs in tqdm(Loader):
            images = images.to(device)
            labels = labels.to(device)
            s_labels = s_labels.to(device)

            # ---- purified labels ----
            logits2 = ema(images)
            c_labels = F.softmax(logits2, dim=1)
            pur = nd(c_labels, labels)   # (B, C)

            # ---- KL divergence ----
            kl = F.kl_div(pur.log(), labels, reduction="batchmean")
            mean_kl += kl.item(); kl_batches += 1

            # ---- update dataset noisy labels directly ----
            pur_np = pur.cpu().numpy()
            for j, subset_idx in enumerate(idxs.numpy()):
                old_label = np.argmax(S_Dataset.train_noisy_labels_s[subset_idx])
                new_label = np.argmax(pur_np[j])
                clean_label = np.argmax(S_Dataset.train_labels[subset_idx])

                S_Dataset.train_noisy_labels[subset_idx] = pur_np[j]
                S_Dataset.noise_or_not[subset_idx] = (new_label == clean_label)

                # flips
                if old_label != new_label:
                    if new_label == clean_label: n2c_flip += 1
                    else:
                        if old_label == clean_label: c2n_flip += 1
                        else: n2n_flip += 1

    # ---- data accuracy ----
    n_total = len(S_Dataset.noise_or_not)
    n_true = np.sum(S_Dataset.noise_or_not)
    data_acc = n_true / n_total

    # ---- logging ----
    json_f["data_acc"].append(data_acc)
    json_f["n2c_flip"].append(n2c_flip)
    json_f["n2n_flip"].append(n2n_flip)
    json_f["c2n_flip"].append(c2n_flip)
    json_f["mean_kl"].append(mean_kl / kl_batches)

    print(f"Purified ({n_total} samples), acc={data_acc:.4f}, correct={n2c_flip}, wrong={n2n_flip}, incorrect={c2n_flip}, mean_kl={mean_kl / kl_batches:.6f}")
    return json_f

with open(txtfile, "a") as myfile:
    myfile.write("epoch: train_acc train_loss meta_acc meta_loss test_acc test_loss \n")

torch.cuda.reset_peak_memory_stats()  # Reset stats

if args.load != None:
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

k_folds = args.n_round  # Number of folds
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

for r in range(args.start_round, args.n_round + 1):
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
        if fold == r - 1:  # Select the r-th fold
            base_dataset = torch.utils.data.Subset(train_dataset, train_idx)
            meta_dataset = torch.utils.data.Subset(train_dataset, val_idx)
            break
    
    # Data Loader (Input Pipeline)
    print("loading dataset...")
    train_loader = torch.utils.data.DataLoader(
        dataset=base_dataset,
        batch_size=batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        shuffle=True,
    )

    meta_loader = torch.utils.data.DataLoader(
        dataset=meta_dataset,
        batch_size=batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        shuffle=False,
    )

    cudnn.benchmark = True

    print("Building model...")
    if args.dataset == "cifar100" and args.noise_rate == 0.7:
        cnn = timm.create_model("convnext_base", pretrained=True, num_classes=num_classes).cuda()
    else:
        cnn = timm.create_model("convnext_small", pretrained=True, num_classes=num_classes).cuda()
    
    cnn_ema = EMA(
        cnn,
        beta=args.beta,
        update_after_step=0,
        update_every=1,
    )

    cnn_ema.update()  # Update EMA
    optimizer = optim.SGD(cnn.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    matching = train_dataset.train_noisy_labels.argmax(1) == train_dataset.train_labels.argmax(1)
    print("Meta_dataset Acc:" + str(matching.mean()))

    # Pre-training
    count = 0
    for epoch in range(1, args.n_epoch + 1):
        cnn.train()
        train_acc, train_loss = train(train_loader, epoch, cnn, optimizer)
        cnn_ema.update()
    
        meta_acc, meta_loss = evaluate(meta_loader, cnn)          # model1 only
        meta_acc_ema, meta_loss_ema = evaluate(meta_loader, cnn_ema)
    
        test_acc, test_loss = evaluate(test_loader, cnn)
        test_acc_ema, test_loss_ema = evaluate(test_loader, cnn_ema)
    
        print(f"Epoch [{epoch}/{args.n_epoch}] Test {len(test_dataset)} imgs | "
              f"Model: {test_acc:.4f}% | EMA: {test_acc_ema:.4f}% | "
              f"loss: {test_loss:.4f} | EMA loss: {test_loss_ema:.4f}")
    
        with open(txtfile, "a") as f:
            f.write(f"{epoch}: {train_acc} {train_loss} "
                    f"{meta_acc} {meta_acc_ema} {meta_loss} {meta_loss_ema} "
                    f"{test_acc} {test_acc_ema} {test_loss} {test_loss_ema}\n")
    
        # Early stopping
        if epoch == 1:
            val_acc_max = meta_acc
        elif meta_acc >= val_acc_max:
            val_acc_max = meta_acc
            count = 0
        else:
            count += 1
    
        if count > args.ES_step:
            break
    
    # --- Noise detector ---
    hidden_dim = 16 if args.dataset == "cifar10" else 32
    nd = NoiseDetector(input_dim=num_classes, hidden_dim=hidden_dim, temperature=args.T).cuda()
    noise_opt = optim.SGD(nd.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

    # --- Meta-training ---
    cnn.train(); cnn_ema.eval(); nd.train()
    meta_train(meta_loader, args.meta_iter, cnn, optimizer, cnn_ema, nd, noise_opt)

    # Data cleansing
    json_f = purify_dataset(cnn, cnn_ema, nd, base_dataset, train_dataset, json_f)
    print(f"Peak memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

    # --- Check stopping condition ---
    mean_kl_now = json_f["mean_kl"][-1]
    if mean_kl_now > min(json_f["mean_kl"]) or r == args.n_round:
        x = np.arange(1, len(json_f["mean_kl"]) + 1)
        y = np.array(json_f["mean_kl"])
    
        c_fixed = y.min()
        x = x[:y.argmin()]
        y = np.log(y[:y.argmin()] - c_fixed)
    
        def exp_func_fixed_c(x, a, b): return np.log(a) - b * x
        def pred_round(a, b, c): return (1 / b) * np.log(a / c)
    
        popt, _ = curve_fit(exp_func_fixed_c, x, y, p0=(0.02, 1))
        a_fit, b_fit = popt
        pred_r = pred_round(a_fit, b_fit, c_fixed)
    
        json_f["pred_opt_r"].append(pred_r)
        opt_r = int(pred_r) + 1
        if r > opt_r: break
    
    # --- Save purified dataset ---
    os.makedirs("./data/purified", exist_ok=True)
    save_path = f"./data/purified/{model_str}_at_round{r}.pt"
    torch.save(train_dataset, save_path)
    torch.cuda.empty_cache()
    
    # --- Save json log ---
    with open(jsonfile, "w") as f: json.dump(json_f, f)