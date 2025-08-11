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


def normalise_for_astropt(x):
    std, mean = torch.std_mean(x, dim=1, keepdim=True)
    return (x - mean) / (std + 1e-8)


def data_transforms():
    return transforms.Compose([transforms.Lambda(normalise_for_astropt)])


def _process_galaxy_wrapper(idx, func, modes):
    """This function ensures that the image is tokenised in the same way as the
    pre-trained model is expecting"""
    mode0_im = np.array(flux_to_pil(idx[f"{modes[0]}_image"])).swapaxes(0, 2)
    mode0_im = func(torch.from_numpy(mode0_im).to(torch.float)).to(torch.float)
    mode0_positions = torch.arange(0, len(mode0_im), dtype=torch.long)
    mode1_im = np.array(flux_to_pil(idx[f"{modes[1]}_image"])).swapaxes(0, 2)
    mode1_im = func(torch.from_numpy(mode1_im).to(torch.float)).to(torch.float)
    mode1_positions = torch.arange(0, len(mode1_im), dtype=torch.long)
    return {
        f"{modes[0]}_images": mode0_im,
        f"{modes[0]}_positions": mode0_positions,
        f"{modes[1]}_images": mode1_im,
        f"{modes[1]}_positions": mode1_positions,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for astronomy")
    parser.add_argument("--mode0", default="hsc", help="First modality name")
    parser.add_argument("--mode1", default="jwst", help="Second modality name")
    parser.add_argument(
        "--input-dataset",
        default="Smith42/jwst_hsc_crossmatched",
        help="Input HuggingFace dataset",
    )
    parser.add_argument(
        "--output-dataset",
        default="UniverseTBD/jwst_hsc_embeddings",
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

    mode0 = args.mode0
    mode1 = args.mode1
    hf_ds = args.input_dataset
    upload_ds = args.output_dataset
    batch_size = args.batch_size

    df = pl.DataFrame()

    for size in ["015M", "095M", "850M"]:
        model = load_astropt("Smith42/astroPT_v2.0", path=f"astropt/{size}").to("cuda")

        galproc = GalaxyImageDataset(
            None,
            spiral=True,
            transform={"images": data_transforms()},
            modality_registry=model.modality_registry,
        )

        ds = (
            load_dataset(hf_ds, split="train", streaming=True)
            .select_columns((f"{mode0}_image", f"{mode1}_image"))
            .map(
                partial(
                    _process_galaxy_wrapper,
                    func=galproc.process_galaxy,
                    mode0=mode0,
                    mode1=mode1,
                )
            )
            .remove_columns((f"{mode0}_image", f"{mode1}_image"))
        )

        dl = iter(DataLoader(ds, batch_size=batch_size, num_workers=32))

        zs_mode0 = []
        zs_mode1 = []
        for B in tqdm(dl):
            B_mode0 = {}
            B_mode0["images"] = B[f"{mode0}_images"].to("cuda")
            B_mode0["images_positions"] = B[f"{mode0}_positions"].to("cuda")
            zs_mode0.append(model.generate_embeddings(B_mode0)["images"].detach())

            B_mode1 = {}
            B_mode1["images"] = B[f"{mode1}_images"].to("cuda")
            B_mode1["images_positions"] = B[f"{mode1}_positions"].to("cuda")
            zs_mode1.append(model.generate_embeddings(B_mode1)["images"].detach())

        zs_mode0 = torch.cat(zs_mode0)
        zs_mode1 = torch.cat(zs_mode1)

        mknn_score = mknn(zs_mode0.cpu().numpy(), zs_mode1.cpu().numpy(), args.knn_k)

        print(f"\nMutual KNN Score: {mknn_score:.4f}")

        df = df.with_columns(
            [
                pl.Series(
                    f"astropt_{size.lstrip('0')}_{mode0}".lower(),
                    zs_mode0.cpu().numpy(),
                ),
                pl.Series(
                    f"astropt_{size.lstrip('0')}_{mode1}".lower(),
                    zs_mode1.cpu().numpy(),
                ),
            ]
        )

    print(df)
    Dataset.from_polars(df).push_to_hub(upload_ds)


if __name__ == "__main__":
    main()
