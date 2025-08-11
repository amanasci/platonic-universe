import argparse
from functools import partial

import numpy as np
import polars as pl
import torch
from astropt.local_datasets import GalaxyImageDataset
from astropt.model_utils import load_astropt
from datasets import Dataset, load_dataset
from PIL import ImageFile
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from pu.pu import flux_to_pil, mknn

ImageFile.LOAD_TRUNCATED_IMAGES = True

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
            im = np.array(flux_to_pil(idx[f"{mode}_image"])).swapaxes(0, 2)
            im = self.galproc.process_galaxy(torch.from_numpy(im).to(torch.float)).to(torch.float)
            result[f"{mode}_images"] = im
            result[f"{mode}_positions"] = torch.arange(0, len(im), dtype=torch.long)
        
        return result


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for astronomy")
    parser.add_argument("--modes", nargs=2, default=["hsc", "jwst"], help="Modality names")
    parser.add_argument(
        "--input-dataset",
        default="Smith42/jwst_hsc_crossmatched",
        help="Input HuggingFace dataset",
    )
    parser.add_argument(
        "--output-dataset",
        default=None,#"UniverseTBD/jwst_hsc_embeddings",
        help="Output HuggingFace dataset",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for processing"
    )
    parser.add_argument(
        "--num-workers", type=int, default=32, help="Number of data loader workers"
    )
    parser.add_argument(
        "--knn-k", type=int, default=10, help="K value for mutual KNN calculation"
    )

    args = parser.parse_args()

    modes = args.modes
    hf_ds = args.input_dataset
    upload_ds = args.output_dataset
    batch_size = args.batch_size

    df = pl.DataFrame()
    for size in ["015M", "095M", "850M"]:
        model = load_astropt("Smith42/astroPT_v2.0", path=f"astropt/{size}").to("cuda")
        model.eval()
        processor = PreprocessAstropt(model.modality_registry, modes)

        ds = (
            load_dataset(hf_ds, split="train", streaming=True)
            .select_columns([f"{mode}_image" for mode in modes])
            .map(processor)
            .remove_columns([f"{mode}_image" for mode in modes])
        )

        dl = iter(DataLoader(ds, batch_size=batch_size, num_workers=32))

        zs = {mode: [] for mode in modes} 
        with torch.no_grad():
            for B in tqdm(dl):
                for mode in modes:
                    inputs = {
                        "images": B[f"{mode}_images"].to("cuda"),
                        "images_positions": B[f"{mode}_positions"].to("cuda")
                    }
                    zs[mode].append(model.generate_embeddings(inputs)["images"].detach())

        zs = {mode: torch.cat(embs) for mode, embs in zs.items()}
        mknn_score = mknn(zs[modes[0]].cpu().numpy(), zs[modes[1]].cpu().numpy(), args.knn_k)

        print(f"\nMutual KNN Score: {mknn_score:.8f}")

        df = df.with_columns(
            [
                pl.Series(
                    f"astropt_{size.lstrip('0')}_{mode}".lower(),
                    embs.cpu().numpy(),
                )
                for mode, embs in zs.items()
            ]
        )

    print(df)
    if upload_dataset is not None:
        Dataset.from_polars(df).push_to_hub(upload_ds)


if __name__ == "__main__":
    main()
