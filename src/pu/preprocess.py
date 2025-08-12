import torch
from astropt.local_datasets import GalaxyImageDataset
from torchvision import transforms

from pu.pu import flux_to_pil


class PreprocessHF:
    """Preprocessor that converts galaxy images to the format expected by Dino and ViT models"""

    def __init__(self, modes, autoproc):
        self.modes = modes
        self.autoproc = autoproc

    def __call__(self, idx):
        result = {}
        for mode in self.modes:
            im = flux_to_pil(idx[f"{mode}_image"], mode, self.modes)
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
            im = flux_to_pil(idx[f"{mode}_image"], mode, self.modes).swapaxes(0, 2)
            im = self.galproc.process_galaxy(torch.from_numpy(im).to(torch.float)).to(
                torch.float
            )
            result[f"{mode}_images"] = im
            result[f"{mode}_positions"] = torch.arange(0, len(im), dtype=torch.long)

        return result
