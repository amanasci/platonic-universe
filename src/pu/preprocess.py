from functools import partial

import numpy as np
import torch
from astropt.local_datasets import GalaxyImageDataset
from torchvision import transforms

from pu.zoom import resize_galaxy_to_fit


class PreprocessHF:
    """Preprocessor that converts galaxy images to the format expected by Dino and ViT models"""

    def __init__(
        self,
        modes,
        autoproc,
        resize=False,
    ):
        self.modes = modes
        self.autoproc = autoproc
        self.f2p = partial(flux_to_pil, resize=resize)

    def __call__(self, idx):
        result = {}
        for mode in self.modes:
            if (mode != "desi") and (mode != "sdss"):
                try:
                    im = flux_to_pil(idx[f"{mode}_image"], mode, self.modes)
                except KeyError:
                    # Assume the dataset does not name the images by modality
                    im = self.f2p(idx["image"], mode, self.modes)
                result[f"{mode}"] = self.autoproc(im, return_tensors="pt")[
                    "pixel_values"
                ].squeeze()

        return result


class PreprocessAstropt:
    """Preprocessor that converts galaxy images to the format expected by AstroPT models"""

    @staticmethod
    def normalise_for_astropt(x):
        std, mean = torch.std_mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + 1e-8)

    @classmethod
    def data_transforms(cls):
        return transforms.Compose([transforms.Lambda(cls.normalise_for_astropt)])

    def __init__(
        self,
        modality_registry,
        modes,
        resize=False,
    ):
        self.galproc = GalaxyImageDataset(
            None,
            spiral=True,
            transform={"images": self.data_transforms()},
            modality_registry=modality_registry,
        )
        self.modes = modes
        self.f2p = partial(flux_to_pil, resize=resize)

    def __call__(self, idx):
        result = {}
        for mode in self.modes:
            if (mode != "desi") and (mode != "sdss"):
                try:
                    im = self.f2p(idx[f"{mode}_image"], mode, self.modes).swapaxes(0, 2)
                except KeyError:
                    # Assume the dataset does not name the images by modality
                    im = flux_to_pil(idx["image"], mode, self.modes).swapaxes(0, 2)
                im = self.galproc.process_galaxy(
                    torch.from_numpy(im).to(torch.float)
                ).to(torch.float)
                result[f"{mode}_images"] = im
                result[f"{mode}_positions"] = torch.arange(0, len(im), dtype=torch.long)

        return result


def flux_to_pil(blob, mode, modes, resize=False, percentile_norm=True):
    """
    Convert raw fluxes to PIL imagery
    """

    def _norm(chan, percentiles=None):
        if percentiles is not None:
            # if percentiles are present norm by them
            v0, v1 = percentiles
            chan = ((chan - v0) / (v1 - v0)).clip(0, 1)
        else:
            # else assume we norm per image
            scale = np.percentile(chan, 99) - np.percentile(chan, 1)
            chan = np.arcsinh((chan - np.percentile(chan, 1)) / scale)
            chan = (chan - chan.min()) / (chan.max() - chan.min())
        return chan

    arr = np.asarray(blob["flux"], np.float32)
    if mode == "hsc":
        if arr.ndim == 3:
            arr = np.stack([arr[0], arr[1], arr[3]], axis=-1)  # grz
        elif arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        else:
            raise ValueError(f"Array shape {arr.shape} for {mode} not recognised")

        if (("jwst" in modes) or ("desi" in modes) or ("sdss" in modes)) and resize:
            # if comparing hsc to jwst resize hsc so it matches hsc
            arr = resize_galaxy_to_fit(
                arr, force_extent=(68, 92, 68, 92), target_size=96
            )

        if percentile_norm:
            norm_consts = {
                "g": (-0.00847, 0.238),
                "r": (-0.0129, 0.480),
                "z": (-0.0291, 1.02),
            }
            arr = np.stack(
                [
                    _norm(arr[..., -1], norm_consts[band])
                    for ii, band in enumerate(("g", "r", "z"))
                ],
                axis=-1,
            )

    if mode == "jwst":  # 0.04 pixel per arcsec
        if arr.ndim == 3:
            arr = np.stack([arr[0], arr[4], arr[6]], axis=-1)
        elif arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        else:
            raise ValueError(f"Array shape {arr.shape} for {mode} not recognised")

        if percentile_norm:
            norm_consts = {
                "f090w": (-0.0741, 2.38),
                "f277w": (-0.0183, 5.76),
                "f444w": (-0.0309, 4.06),
            }
            arr = np.stack(
                [
                    _norm(arr[..., -1], norm_consts[band])
                    for ii, band in enumerate(("f090w", "f277w", "f444w"))
                ],
                axis=-1,
            )

    if mode == "legacysurvey":
        if arr.ndim == 3:
            arr = np.stack([arr[0], arr[1], arr[3]], axis=-1)  # grz
        elif arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        else:
            raise ValueError(f"Array shape {arr.shape} for {mode} not recognised")

        if resize:
            # we always resize legacy to match hsc for our use-case
            arr = resize_galaxy_to_fit(
                arr, force_extent=(36, 124, 36, 124), target_size=160
            )

        if percentile_norm:
            norm_consts = {
                "g": (-0.00216, 0.000766),
                "r": (-0.00317, 0.0173),
                "z": (-0.00843, 0.0350),
            }
            arr = np.stack(
                [
                    _norm(arr[..., -1], norm_consts[band])
                    for ii, band in enumerate(("g", "r", "z"))
                ],
                axis=-1,
            )

    if not percentile_norm:
        arr = _norm(arr)
    arr = (arr * 255).astype(np.uint8)

    return arr
