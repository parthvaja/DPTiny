"""Train the MNIST CNN baseline on Fashion-MNIST.

The model is the one in examples/mnist_cnn.py, unchanged. Images are
standardized with the training-set mean and standard deviation, and the run
is seeded. Trains for 10 epochs with Adam (lr 0.001, batch size 64), reports
loss and accuracy after each epoch, then the confusion matrix, per-class
accuracy and the most confused pair of classes.
"""

import time

import numpy as np

from dptiny import (
    Variable,
    is_available,
    is_gpu,
    no_grad,
    softmax_cross_entropy,
    test_mode,
    to_cpu,
    to_gpu,
    use_gpu,
    xp,
)
from dptiny.data import DataLoader
from dptiny.data.fashion_mnist import CLASSES, get_fashion_mnist
from dptiny.nn import (
    Conv2d,
    Dropout,
    Flatten,
    Linear,
    MaxPool2d,
    ReLU,
    Sequential,
)
from dptiny.optim import Adam

SEED = 42
BATCH_SIZE = 64
MAX_EPOCH = 10
LR = 0.001
NUM_CLASSES = len(CLASSES)

if is_available():
    try:
        use_gpu()
        print("GPU enabled for training.")
    except RuntimeError as exc:
        print(f"GPU not usable ({exc}); training on CPU.")
else:
    print("GPU not available; training on CPU.")

# Seed both NumPy and the active backend (CuPy on GPU). Weight init, dropout
# masks and DataLoader shuffling all draw from xp.random.
np.random.seed(SEED)
xp.random.seed(SEED)

print("Loading Fashion-MNIST dataset...")
X_train, X_test, y_train, y_test = get_fashion_mnist(flatten=False)

# Standardize with statistics of the TRAINING set only (no test leakage).
mean = float(X_train.mean())
std = float(X_train.std())
X_train = ((X_train - mean) / std).astype(np.float32)
X_test = ((X_test - mean) / std).astype(np.float32)
print(f"Train mean = {mean:.4f}, std = {std:.4f} (used for both splits)")

if is_gpu():
    X_train = to_gpu(X_train)
    X_test = to_gpu(X_test)
    y_train = to_gpu(y_train)
    y_test = to_gpu(y_test)

# Baseline model from examples/mnist_cnn.py, unchanged.
model = Sequential(
    Conv2d(1, 16, 3, pad=1),
    ReLU(),
    MaxPool2d(2),
    Conv2d(16, 32, 3, pad=1),
    ReLU(),
    MaxPool2d(2),
    Flatten(),
    Linear(32 * 7 * 7, 128),
    ReLU(),
    Dropout(0.3),
    Linear(128, 10),
)
if is_gpu():
    model.to_gpu()

train_loader = DataLoader((X_train, y_train), BATCH_SIZE)
test_loader = DataLoader((X_test, y_test), BATCH_SIZE, shuffle=False)
optimizer = Adam(model, lr=LR)


def predict(loader):
    """Return (true labels, predicted labels) as NumPy arrays."""
    trues, preds = [], []
    with test_mode(), no_grad():
        for x, t in loader:
            y = model(Variable(x))
            preds.append(to_cpu(y.data.argmax(axis=1)))
            trues.append(to_cpu(t))
    return np.concatenate(trues), np.concatenate(preds)


start = time.time()
for epoch in range(1, MAX_EPOCH + 1):
    model.train()
    sum_loss, correct, count = 0.0, 0, 0
    for x, t in train_loader:
        y = model(Variable(x))
        loss = softmax_cross_entropy(y, t)

        model.cleargrads()
        loss.backward()
        optimizer.update()

        n = len(t)
        sum_loss += float(loss.data) * n
        correct += int((y.data.argmax(axis=1) == t).sum())
        count += n

    true, pred = predict(test_loader)
    test_acc = float((true == pred).mean())
    print(
        f"epoch {epoch:2d}/{MAX_EPOCH} | train loss {sum_loss / count:.4f} | "
        f"train acc {correct / count:.4f} | test acc {test_acc:.4f} | "
        f"{time.time() - start:.1f}s"
    )

print(f"\nTraining completed in {time.time() - start:.1f} seconds")

# Confusion matrix: rows = true class, columns = predicted class.
true, pred = predict(test_loader)
cm = np.bincount(
    true * NUM_CLASSES + pred, minlength=NUM_CLASSES * NUM_CLASSES
).reshape(NUM_CLASSES, NUM_CLASSES)

print("\nConfusion matrix (rows = true, columns = predicted):")
print("     " + " ".join(f"{j:>5d}" for j in range(NUM_CLASSES)))
for i in range(NUM_CLASSES):
    print(f"{i:>3d}  " + " ".join(f"{v:>5d}" for v in cm[i]))

print("\nPer-class accuracy:")
per_class = cm.diagonal() / cm.sum(axis=1)
for i, name in enumerate(CLASSES):
    print(f"  {i} {name:<12s} {per_class[i]:.4f}  ({cm[i, i]}/{cm[i].sum()})")
print(f"Overall test accuracy: {cm.trace() / cm.sum():.4f}")

# Most confused pair: count mistakes in both directions (i->j plus j->i).
pair = cm + cm.T
np.fill_diagonal(pair, 0)
a, b = sorted(np.unravel_index(int(pair.argmax()), pair.shape))
print(
    f"\nMost confused pair: {CLASSES[a]} <-> {CLASSES[b]} "
    f"({pair[a, b]} test images: {cm[a, b]} {CLASSES[a]} -> {CLASSES[b]}, "
    f"{cm[b, a]} {CLASSES[b]} -> {CLASSES[a]})"
)
