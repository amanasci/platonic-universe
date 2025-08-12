import numpy as np
from sklearn.neighbors import NearestNeighbors

from pu.zoom import resize_galaxy_to_fit


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


def flux_to_pil(blob, mode):
    """
    Convert raw fluxes to PIL imagery
    """

    def _norm(chan):
        chan = np.arcsinh(chan / np.percentile(chan, 95))
        chan = (chan - chan.min()) / (chan.max() - chan.min())
        return chan

    arr = np.asarray(blob["flux"], np.float32)
    if mode == "hsc":
        if arr.ndim == 3:
            arr = np.stack([arr[0], arr[1], arr[2]], axis=-1)  # gri
        elif arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        arr, _ = resize_galaxy_to_fit(arr)
    if mode == "jwst":
        if arr.ndim == 3:
            arr = np.stack([arr[1], arr[3], arr[6]], axis=-1)  #
        elif arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
    if mode == "legacysurvey":
        if arr.ndim == 3:
            arr = np.stack([arr[0], arr[1], arr[3]], axis=-1)  # grz
        elif arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        arr, _ = resize_galaxy_to_fit(arr)

    arr = np.stack([_norm(arr[..., ii]) for ii in range(3)], axis=-1)
    arr = (arr * 255).astype(np.uint8)

    return arr
