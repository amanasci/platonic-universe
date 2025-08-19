import argparse

import numpy as np
import polars as pl
import torch
from astropt.model_utils import load_astropt
from datasets import Dataset, load_dataset, concatenate_datasets
from PIL import ImageFile
from torch.utils.data import DataLoader
from tqdm import tqdm

from pu.preprocess import PreprocessAstropt
from pu.metrics import mknn

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for astronomy")
    parser.add_argument(
        "--mode", default="jwst", help="Mode to compare to hsc"
    )
    parser.add_argument(
        "--output-dataset",
        default=None,  # "UniverseTBD/jwst_hsc_embeddings",
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

    comp_mode = args.mode
    modes = ["hsc", comp_mode]
    hf_ds = f"Smith42/{comp_mode}_hsc_crossmatched"
    upload_ds = args.output_dataset
    batch_size = args.batch_size

    def filterfun(idx):
        if "jwst" != comp_mode:
            return True
        else:
            im = idx["jwst_image"]["flux"][3]
            v0, v1 = np.nanpercentile(im, 5), np.nanpercentile(im, 99)
            if v0 - v1 == 0:
                return False
            else:
                return True

    df = pl.DataFrame()
    for size in ["015M", "095M", "850M"]:
        model = load_astropt("Smith42/astroPT_v2.0", path=f"astropt/{size}").to("cuda")
        model.eval()
        processor = PreprocessAstropt(model.modality_registry, modes, resize=False)

        if (comp_mode == "jwst") or (comp_mode == "legacysurvey"):
            ds = (
                load_dataset(hf_ds, split="train", streaming=True)
                .select_columns([f"{mode}_image" for mode in modes])
                .filter(filterfun)
                .map(processor)
                .remove_columns([f"{mode}_image" for mode in modes])
            )
        elif comp_mode == "sdss":
            ds = (concatenate_datasets((
                load_dataset(hf_ds, split="train", streaming=True),
                load_dataset("Shashwat20/SDSS_Interpolated", split="train", streaming=True)
            ), axis=1)
                .rename_column("image", "hsc_image")
                .select_columns(["hsc_image", "embedding"])
                .filter(filterfun)
                .map(processor)
                .remove_columns(["hsc_image"])
            )
        elif comp_mode == "desi":
            ds = (concatenate_datasets((
                load_dataset(hf_ds, split="train", streaming=True),
                load_dataset("Smith42/specformer_desi", split="train", streaming=True)
            ), axis=1)
                .rename_column("image", "hsc_image")
                .select_columns(["hsc_image", "embeddings"])
                .filter(filterfun)
                .map(processor)
                .remove_columns(["hsc_image"])
            )
        else:
            raise NotImplementedError


        dl = iter(DataLoader(ds, batch_size=batch_size))

        zs = {mode: [] for mode in modes}
        with torch.no_grad():
            for B in tqdm(dl):
                for mode in modes:
                    if mode == "sdss":
                        zs[mode].append(torch.tensor(np.array(B["embedding"])).T)
                    elif mode == "desi":
                        zs[mode].append(torch.tensor(np.array(B["embeddings"])).T)
                    else:
                        inputs = {
                            "images": B[f"{mode}_images"].to("cuda"),
                            "images_positions": B[f"{mode}_positions"].to("cuda"),
                        }
                        zs[mode].append(
                            model.generate_embeddings(inputs)["images"].detach()
                        )

        zs = {mode: torch.cat(embs) for mode, embs in zs.items()}
        mknn_score = mknn(
            zs[modes[0]].cpu().numpy(), zs[modes[1]].cpu().numpy(), args.knn_k
        )

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

    #print(df)
    if upload_ds is not None:
        Dataset.from_polars(df).push_to_hub(upload_ds)


if __name__ == "__main__":
    main()
