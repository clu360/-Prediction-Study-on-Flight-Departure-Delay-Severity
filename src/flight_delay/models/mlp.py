"""Deterministic minibatch training for the Phase IV factorial experiment."""

import hashlib
import os

import numpy as np
import torch
from torch import nn


def training_weights(labels):
    if not np.isin(labels, [0, 1, 2]).all():
        raise ValueError("Invalid training labels")
    counts = np.bincount(labels, minlength=3)
    if (counts == 0).any():
        raise ValueError("Training window is missing a class")
    return (len(labels) / (3 * counts)).astype(np.float32)


def create(input_width, candidate, seed, threads, device):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    first, second = candidate["widths"]
    return nn.Sequential(
        nn.Linear(input_width, first), nn.ReLU(), nn.Dropout(candidate["dropout"]),
        nn.Linear(first, second), nn.ReLU(), nn.Linear(second, 3),
    ).to(device)


def state_sha(model):
    sha = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        values = tensor.detach().cpu().contiguous().numpy()
        sha.update(repr((name, values.shape, values.dtype.str)).encode())
        sha.update(values.tobytes())
    return sha.hexdigest()


def tensor_batch(matrix, indices, device):
    return torch.from_numpy(matrix[indices].toarray()).to(device)


@torch.inference_mode()
def predict(model, matrix, batch_size, device):
    model.eval()
    return np.concatenate([
        model(tensor_batch(matrix, slice(i, i + batch_size), device))
        .softmax(dim=1).cpu().numpy()
        for i in range(0, matrix.shape[0], batch_size)
    ])


@torch.inference_mode()
def validation_loss(model, matrix, labels, batch_size, device):
    model.eval()
    total = 0.0
    for i in range(0, len(labels), batch_size):
        sl = slice(i, i + batch_size)
        logits = model(tensor_batch(matrix, sl, device))
        y = torch.as_tensor(labels[sl].astype(np.int64), device=device)
        total += nn.functional.cross_entropy(logits, y, reduction="sum").item()
    return total / len(labels)


def train(matrix, labels, candidate, settings, seed, device, epochs, *,
          weighted=False, stop=None, on_epoch=None):
    model = create(matrix.shape[1], candidate, seed, settings["threads"], device)
    counts = training_weights(labels)
    weights = torch.from_numpy(counts).to(device) if weighted else None
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=candidate["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    rng = np.random.default_rng(seed)
    batch_size = settings["batch_size"]
    best_epoch, best_loss = 0, float("inf")
    curves = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(labels))
        total = 0.0
        denominator = 0.0
        for offset in range(0, len(labels), batch_size):
            indices = order[offset:offset + batch_size]
            y = torch.as_tensor(labels[indices].astype(np.int64), device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(tensor_batch(matrix, indices, device)), y)
            loss.backward()
            optimizer.step()
            batch_weight = float(counts[labels[indices]].sum()) if weighted else len(indices)
            total += float(loss.detach().item()) * batch_weight
            denominator += batch_weight
        record = {"epoch": epoch, "train_loss": total / denominator}
        if stop is not None:
            record["stop_loss"] = validation_loss(model, *stop, batch_size, device)
            if record["stop_loss"] < best_loss:
                best_loss, best_epoch = record["stop_loss"], epoch
        if not all(np.isfinite(value) for value in record.values()):
            raise ValueError(f"Nonfinite training diagnostics at epoch {epoch}")
        curves.append(record)
        if on_epoch is not None:
            on_epoch(record)
        if stop is not None and epoch - best_epoch >= settings["patience"]:
            break
    return model, best_epoch if stop is not None else epochs, curves

