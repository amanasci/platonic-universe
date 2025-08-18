import torch
from astropt.local_datasets import GalaxyImageDataset
from torchvision import transforms
import numpy as np

from pu.zoom import resize_galaxy_to_fit

class PreprocessHF:
    """Preprocessor that converts galaxy images to the format expected by Dino and ViT models"""

    def __init__(self, modes, autoproc):
        self.modes = modes
        self.autoproc = autoproc

    def __call__(self, idx):
        result = {}
        for mode in self.modes:
            if (mode != "desi") and (mode != "sdss"): 
                try:
                    im = flux_to_pil(idx[f"{mode}_image"], mode, self.modes)
                except KeyError as e:
                    # Assume the dataset does not name the images by modality
                    im = flux_to_pil(idx["image"], mode, self.modes)
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

    def __init__(self, modality_registry, modes):
        self.galproc = GalaxyImageDataset(
            None,
            spiral=True,
            transform={"images": self.data_transforms()},
            modality_registry=modality_registry,
        )
        self.modes = modes

    def __call__(self, idx):
        result = {}
        for mode in self.modes:
            if (mode != "desi") and (mode != "sdss"): 
                try:
                    im = flux_to_pil(idx[f"{mode}_image"], mode, self.modes).swapaxes(0, 2)
                except KeyError as e:
                    # Assume the dataset does not name the images by modality
                    im = flux_to_pil(idx[f"image"], mode, self.modes).swapaxes(0, 2)
                im = self.galproc.process_galaxy(torch.from_numpy(im).to(torch.float)).to(
                    torch.float
                )
                result[f"{mode}_images"] = im
                result[f"{mode}_positions"] = torch.arange(0, len(im), dtype=torch.long)

        return result


def flux_to_pil(blob, mode, modes):
    """
    Convert raw fluxes to PIL imagery
    """

    def _norm(chan):
        scale = np.percentile(chan, 99) - np.percentile(chan, 1)
        chan = np.arcsinh((chan - np.percentile(chan, 1)) / scale)
        #v0, v1 = np.nanpercentile(chan, 5), np.nanpercentile(chan, 99)
        #chan = ((chan - v0) / (v1 - v0)).clip(0, 1) * 255
        chan = (chan - chan.min()) / (chan.max() - chan.min())
        return chan

    arr = np.asarray(blob["flux"], np.float32)
    if mode == "hsc":
        if arr.ndim == 3:
            arr = np.stack([arr[2], arr[2], arr[2]], axis=-1)  # gri
        elif arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if ("jwst" in modes) or ("desi" in modes) or ("sdss" in modes):
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

    arr = _norm(arr)#np.stack([_norm(arr[..., ii]) for ii in range(3)], axis=-1)
    arr = (arr * 255).astype(np.uint8)

    return arr
