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


def flux_to_pil(blob, mode, modes):
    """
    Convert raw fluxes to PIL imagery
    """

    def _norm(chan):
        scale = np.percentile(chan, 99) - np.percentile(chan, 1)
        chan = np.arcsinh((chan - np.percentile(chan, 1)) / scale)
        chan = (chan - chan.min()) / (chan.max() - chan.min())
        return chan

    arr = np.asarray(blob["flux"], np.float32)
    if mode == "hsc":
        if arr.ndim == 3:
            arr = np.stack([arr[0], arr[1], arr[2]], axis=-1)  # gri
        elif arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if "jwst" in modes:
            # if comparing hsc to jwst resize hsc so it matches hsc
            arr = resize_galaxy_to_fit(
                arr, force_extent=(68, 92, 68, 92), target_size=96
            )
    if (
        mode == "jwst"
    ):  # 0.04 pixel per arcsec https://dawn-cph.github.io/dja/blog/2023/07/18/image-data-products/
        if arr.ndim == 3:
            arr = np.stack([arr[1], arr[3], arr[6]], axis=-1)  #
        elif arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
    if mode == "legacysurvey":
        if arr.ndim == 3:
            arr = np.stack([arr[0], arr[1], arr[3]], axis=-1)  # grz
        elif arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        # we always resize legacy to match hsc for our use-case
        arr = resize_galaxy_to_fit(
            arr, force_extent=(36, 124, 36, 124), target_size=160
        )

    arr = np.stack([_norm(arr[..., ii]) for ii in range(3)], axis=-1)
    arr = (arr * 255).astype(np.uint8)

    return arr
