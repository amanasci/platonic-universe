import argparse

import numpy as np
import polars as pl
import torch
from datasets import Dataset, load_dataset
from PIL import ImageFile
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from pu.preprocess import PreprocessHF
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
    for model_name, size in zip(
        [
            "google/vit-base-patch16-224-in21k",
            "google/vit-large-patch16-224-in21k",
            "google/vit-huge-patch14-224-in21k",
        ],
        ["base", "large", "huge"],
    ):
        model = AutoModel.from_pretrained(model_name).to("cuda")
        model.eval()
        processor = PreprocessHF(modes, AutoImageProcessor.from_pretrained(model_name))

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
                    inputs = B[f"{mode}"].to("cuda")
                    zs[mode].append(
                        model(inputs).last_hidden_state.mean(dim=1).detach()
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
    if upload_ds is not None:
        Dataset.from_polars(df).push_to_hub(upload_ds)


if __name__ == "__main__":
    main()
