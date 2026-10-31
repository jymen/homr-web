# Trains the tab digit classifier (src/tab/digits.ts): ten digits and an
# eleventh class, not a digit, over MNIST-shaped 28 x 28 crops (ink 1 on 0).
# Reads the crops tools/tab-digits-data.ts wrote, adds MNIST digits under a
# pen-on-tab augmentation, trains on the CPU with every seed fixed, and exports
# ONNX: input "x" [n, 1, 28, 28] float32, output "logits" [n, 11] float32,
# opset 17. Run through tools/train-tab-digits.sh, which sets up the venv and
# the data; it prints the held-out accuracy per class and the exported file.
#
# usage: python tools/train-tab-digits.py <data dir> <out.onnx> [steps]
import glob
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

SEED = 20261031
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)
torch.set_num_threads(4)

data_dir, out_path = sys.argv[1], sys.argv[2]
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
NOT_A_DIGIT = 10


def normalise(ink):
    """ink: H x W, 0..1. Tight box, longer side 20, centred in 28 x 28: digitInput's form."""
    ys, xs = np.nonzero(ink > 0.5)
    if len(ys) == 0:
        return np.zeros((28, 28), np.float32)
    ink = ink[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    h, w = ink.shape
    s = 20 / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    t = F.interpolate(torch.tensor(ink, dtype=torch.float32)[None, None], size=(nh, nw), mode="area")[0, 0].numpy()
    out = np.zeros((28, 28), np.float32)
    y0, x0 = (28 - nh) // 2, (28 - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = t
    return out


def augment(img, rng):
    """img: 28 x 28 uint8 MNIST. A binarised pen mark: slanted, sheared, thinner or bolder, a line through it."""
    t = torch.tensor(img, dtype=torch.float32)[None, None] / 255
    t = torchvision.transforms.functional.affine(
        t,
        angle=rng.uniform(-20, 20),
        translate=[0, 0],
        scale=1.0,
        shear=[rng.uniform(-25, 15)],
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
    )
    t = F.interpolate(t, size=(28, max(8, round(28 * rng.uniform(0.75, 1.3)))), mode="bilinear")
    k = rng.choice([0, 0, 1, 1, 2])
    if k:
        t = F.max_pool2d(t, 2 * k + 1, stride=1, padding=k)
    elif rng.random() < 0.2:
        t = -F.max_pool2d(-t, 3, stride=1, padding=1)
    a = (t[0, 0].numpy() > rng.uniform(0.3, 0.6)).astype(np.float32)
    if rng.random() < 0.4:  # what is left of a tab line through the mark
        y = rng.randint(int(a.shape[0] * 0.3), int(a.shape[0] * 0.7))
        x0 = rng.randint(0, a.shape[1] // 4)
        x1 = rng.randint(3 * a.shape[1] // 4, a.shape[1])
        a[y : y + rng.randint(1, 2), x0:x1] = 1
    return normalise(a)


def fragment(img, rng):
    """Not a digit: the top, bottom, left or right of an augmented digit, as line erasure or a stem leaves one."""
    a = augment(img, rng)
    ys, xs = np.nonzero(a)
    if len(ys) < 10:
        return None
    if rng.random() < 0.5:
        cut = rng.randint(ys.min() + 3, max(ys.min() + 4, ys.max() - 6))
        keep = (np.arange(28)[:, None] < cut) if rng.random() < 0.5 else (np.arange(28)[:, None] > cut + 6)
    else:
        cut = rng.randint(xs.min() + 3, max(xs.min() + 4, xs.max() - 6))
        keep = (np.arange(28)[None, :] < cut) if rng.random() < 0.5 else (np.arange(28)[None, :] > cut + 6)
    return normalise(a * keep)


def arc(rng):
    """Not a digit: one curved stroke, a slur's or a bracket's (never straight: a hand's 1 is one)."""
    a = np.zeros((40, 40), np.float32)
    rad = rng.uniform(8, 16)
    t0, span = rng.uniform(0, 2 * np.pi), rng.uniform(1.5, 4.0)
    squash = rng.uniform(0.6, 1.0)
    for t in np.linspace(t0, t0 + span, 200):
        x, y = int(20 + rad * np.cos(t)), int(20 + rad * squash * np.sin(t))
        a[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2] = 1
    return normalise(a)


def load(pattern):
    xs, ys = [], []
    for f in sorted(glob.glob(f"{data_dir}/{pattern}-x.f32")):
        xs.append(np.fromfile(f, np.float32).reshape(-1, 28, 28))
        ys.append(np.fromfile(f[: -len("-x.f32")] + "-y.u8", np.uint8))
    return np.concatenate(xs), np.concatenate(ys).astype(np.int64)


class Net(nn.Module):
    """About 117 000 weights, under half a megabyte in fp32."""

    def __init__(self):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(1, 16, 3, padding=1), nn.Conv2d(16, 16, 3, padding=1)
        self.c3, self.c4 = nn.Conv2d(16, 32, 3, padding=1), nn.Conv2d(32, 32, 3, padding=1)
        self.f1, self.f2 = nn.Linear(32 * 7 * 7, 64), nn.Linear(64, 11)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c2(F.relu(self.c1(x)))), 2)
        x = F.max_pool2d(F.relu(self.c4(F.relu(self.c3(x)))), 2)
        return self.f2(self.drop(F.relu(self.f1(x.flatten(1)))))


mnist = torchvision.datasets.MNIST(data_dir, train=True, download=False)
mnist_test = torchvision.datasets.MNIST(data_dir, train=False, download=False)
gen_x, gen_y = load("synthetic-train-*")
fix_x, fix_y = load("fixtures")
test_x, test_y = load("synthetic-test")
print(f"synthetic {len(gen_y)}, fixtures {len(fix_y)}, held-out synthetic {len(test_y)}", flush=True)


def batch(n, rng):
    xs, ys = [], []
    while len(xs) < n:
        r = rng.random()
        if r < 0.55:
            i = rng.randrange(len(gen_y))
            xs.append(gen_x[i]), ys.append(gen_y[i])
        elif r < 0.65:
            i = rng.randrange(len(fix_y))
            xs.append(fix_x[i]), ys.append(fix_y[i])
        elif r < 0.71:
            a = fragment(mnist.data[rng.randrange(len(mnist))].numpy(), rng) if rng.random() < 0.6 else arc(rng)
            if a is not None:
                xs.append(a), ys.append(NOT_A_DIGIT)
        else:
            i = rng.randrange(len(mnist))
            xs.append(augment(mnist.data[i].numpy(), rng)), ys.append(int(mnist.targets[i]))
    return torch.tensor(np.stack(xs))[:, None], torch.tensor(np.array(ys, np.int64))


def accuracy(net, x, y):
    with torch.no_grad():
        guess = net(torch.tensor(x)[:, None]).argmax(1).numpy()
    per = [f"{c}:{(guess[y == c] == c).mean():.3f}" for c in range(11) if (y == c).any()]
    return (guess == y).mean(), " ".join(per)


rng = random.Random(SEED)
net = Net()
opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps)
for step in range(steps):
    net.train()
    x, y = batch(128, rng)
    loss = F.cross_entropy(net(x), y, label_smoothing=0.05)
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
    if step % 1000 == 0 or step == steps - 1:
        net.eval()
        acc, _ = accuracy(net, test_x, test_y)
        print(f"step {step} loss {loss.item():.3f} held-out synthetic {acc:.4f}", flush=True)

net.eval()
acc, per = accuracy(net, test_x, test_y)
print(f"held-out synthetic crops: {acc:.4f} ({per})")
rng_test = random.Random(SEED + 1)
idx = rng_test.sample(range(len(mnist_test)), 5000)
mx = np.stack([augment(mnist_test.data[i].numpy(), rng_test) for i in idx])
my = np.array([int(mnist_test.targets[i]) for i in idx])
acc, per = accuracy(net, mx, my)
print(f"MNIST test digits, augmented: {acc:.4f} ({per})")
acc, per = accuracy(net, fix_x, fix_y)
print(f"fixture crops (seen in training): {acc:.4f} ({per})")
torch.onnx.export(
    net,
    torch.zeros(1, 1, 28, 28),
    out_path,
    input_names=["x"],
    output_names=["logits"],
    dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}},
    opset_version=17,
    dynamo=False,
)
print(f"exported {out_path}")
