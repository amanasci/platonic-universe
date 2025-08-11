from sklearn.neighbors import NearestNeighbors
import numpy as np
from PIL import Image


def mknn(Z1, Z2, k=10):
    """
    Calculate mutual k nearest neighbours
    """
    assert len(Z1) == len(Z2)

    nn1 = (
        NearestNeighbors(n_neighbors=k, metric="cosine")
        .fit(Z1)
        .kneighbors(return_distance=False)
    )
    nn2 = (
        NearestNeighbors(n_neighbors=k, metric="cosine")
        .fit(Z2)
        .kneighbors(return_distance=False)
    )

    overlap = [len(set(a).intersection(b)) for a, b in zip(nn1, nn2)]

    return np.mean(overlap) / k


def flux_to_pil(blob):
    """
    Convert raw fluxes to PIL imagery
    """
    arr = np.asarray(blob["flux"], np.float32)
    if arr.ndim == 3:
        arr = arr[arr.shape[0] // 2]  # middle band
    v0, v1 = np.nanpercentile(arr, 5), np.nanpercentile(arr, 99)
    img = ((arr - v0) / (v1 - v0)).clip(0, 1) * 255
    img = img.astype(np.uint8)
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    return Image.fromarray(img)
