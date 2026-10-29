"""Utilities for CT reconstruction."""
import numpy as np


def normalize_np(x):
    x = np.array(x, copy=True)
    x -= np.min(x)
    maximum = np.max(x)
    if maximum > 0:
        x /= maximum
    return x


def clear(x, normalize=True):
    x = x.detach().cpu().squeeze().numpy()
    return normalize_np(x) if normalize else x


def batchfy(tensor, batch_size):
    # Divide the slices into n // batch_size + 1 contiguous chunks.
    n = len(tensor)
    num_batches = n // batch_size + 1
    return tensor.chunk(num_batches, dim=0)
