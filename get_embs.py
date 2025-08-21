import argparse

import numpy as np
import polars as pl
import torch
from datasets import Dataset, load_dataset, concatenate_datasets
from PIL import ImageFile
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from pu.preprocess import PreprocessHF
from pu.metrics import mknn

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for astronomy")
    parser.add_argument(
        "--mode", default="jwst", help="Mode to compare to hsc"
    )
    parser.add_argument(
        "--model", default="vit", help="Model to run inference on"
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
        "--num-workers", type=int, default=0, help="Number of data loader workers"
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

    if args.model == "vit":
        sizes = ["base", "large", "huge"]
        model_names = [
            "google/vit-base-patch16-224-in21k",
            "google/vit-large-patch16-224-in21k",
            "google/vit-huge-patch14-224-in21k",
        ]
    elif args.model == "dino":
        sizes = ["small", "base", "large", "giant"
        model_names = [f"facebook/dinov2-{size}" for size in sizes]
    elif args.model == "convnext":
        sizes = ["nano", "tiny", "base", "large"]
        model_names = [f"facebook/convnextv2-{size}-22k-224" for size in sizes]
    elif args.model == "ijepa":
        sizes = ["huge", "giant"]
        model_names = [
            "facebook/ijepa_vith14_22k",
            "facebook/ijepa_vitg16_22k",
        ]
    else:
        raise NotImplementedError
        
    df = pl.DataFrame()
    for size, model_name in zip(sizes, model_names):
        model = AutoModel.from_pretrained(model_name).to("cuda")
        model.eval()
        processor = PreprocessHF(modes, AutoImageProcessor.from_pretrained(model_name), resize=False)

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


        dl = iter(DataLoader(ds, batch_size=batch_size, num_workers=args.num_workers))

        zs = {mode: [] for mode in modes}
        with torch.no_grad():
            for B in tqdm(dl):
                for mode in modes:
                    if mode == "sdss":
                        zs[mode].append(torch.tensor(np.array(B["embedding"])).T)
                    elif mode == "desi":
                        zs[mode].append(torch.tensor(np.array(B["embeddings"])).T)
                    else:
                        inputs = B[f"{mode}"].to("cuda")
                        zs[mode].append(
                        # TODO: change hidden state processing per model.
                            model(inputs).last_hidden_state[:, 1:].mean(dim=1).detach()
                        )

        zs = {mode: torch.cat(embs) for mode, embs in zs.items()}
        mknn_score = mknn(
            zs[modes[0]].cpu().numpy(), zs[modes[1]].cpu().numpy(), args.knn_k
        )

        print(f"\nMutual KNN Score: {mknn_score:.8f}")

        df = df.with_columns(
            [
                pl.Series(
                    f"{args.model}_{size.lstrip('0')}_{mode}".lower(),
                    embs.cpu().numpy(),
                )
                for mode, embs in zs.items()
            ]
        )

    #print(df)
    df.write_parquet(f"data/{comp_mode}_{args.model}.parquet")
    if upload_ds is not None:
        Dataset.from_polars(df).push_to_hub(upload_ds)


if __name__ == "__main__":
    main()
