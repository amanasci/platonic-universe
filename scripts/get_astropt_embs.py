import argparse

import numpy as np
import polars as pl
import torch
from astropt.model_utils import load_astropt
from datasets import Dataset, load_dataset
from PIL import ImageFile
from torch.utils.data import DataLoader
from tqdm import tqdm

from pu.preprocess import PreprocessAstropt
from pu.pu import mknn

ImageFile.LOAD_TRUNCATED_IMAGES = True


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for astronomy")
    parser.add_argument(
        "--modes", nargs=2, default=["hsc", "jwst"], help="Modality names"
    )
    parser.add_argument(
        "--input-dataset",
        default="Smith42/jwst_hsc_crossmatched",
        help="Input HuggingFace dataset",
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

    modes = args.modes
    hf_ds = args.input_dataset
    upload_ds = args.output_dataset
    batch_size = args.batch_size

    def filterfun(idx):
        if "jwst" not in modes:
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
        processor = PreprocessAstropt(model.modality_registry, modes)

        ds = (
            load_dataset(hf_ds, split="train", streaming=True)
            .filter(filterfun)
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

    print(df)
    if upload_ds is not None:
        Dataset.from_polars(df).push_to_hub(upload_ds)


if __name__ == "__main__":
    main()
