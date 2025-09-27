import os
import numpy as np
import polars as pl
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from pu.models import get_adapter
from pu.datasets import get_dataset_adapter
from pu.metrics import mknn

def run_experiment(model_alias, mode, output_dataset=None, batch_size=128, num_workers=0, knn_k=10):
    """Runs the embedding generation experiment based on the provided arguments."""

    comp_mode = mode
    modes = ["hsc", comp_mode]
    hf_ds = f"Smith42/{comp_mode}_hsc_crossmatched"
    upload_ds = output_dataset
    batch_size = batch_size

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

    model_map = {
        "vit": (
            ["base", "large", "huge"],
            [
                "google/vit-base-patch16-224-in21k",
                "google/vit-large-patch16-224-in21k",
                "google/vit-huge-patch14-224-in21k",
            ],
        ),
        "dino": (
            ["small", "base", "large", "giant"],
            [f"facebook/dinov2-with-registers-{s}" for s in ["small", "base", "large", "giant"]],
        ),
        "convnext": (
            ["nano", "tiny", "base", "large"],
            [f"facebook/convnextv2-{s}-22k-224" for s in ["nano", "tiny", "base", "large"]],
        ),
        "ijepa": (
            ["huge", "giant"],
            ["facebook/ijepa_vith14_22k", "facebook/ijepa_vitg16_22k"],
        ),
        "astropt": (
            ["015M", "095M", "850M"],
            [f"Smith42/astroPT_v2.0" for _ in range(3)],
        ),
    }

    try:
        sizes, model_names = model_map[model_alias]
    except KeyError:
        raise NotImplementedError(f"Model '{model_alias}' not implemented.")

    df = pl.DataFrame()
    adapter_cls = get_adapter(model_alias)
    for size, model_name in zip(sizes, model_names):
        adapter = adapter_cls(model_name, size, alias=model_alias)
        adapter.load()
        processor = adapter.get_preprocessor(modes)

        # Use dataset adapter to prepare the dataset (centralises dataset-specific logic)
        dataset_adapter_cls = get_dataset_adapter(comp_mode)
        dataset_adapter = dataset_adapter_cls(hf_ds, comp_mode)
        dataset_adapter.load()
        ds = dataset_adapter.prepare(processor, modes, filterfun)


        dl = iter(DataLoader(ds, batch_size=batch_size, num_workers=num_workers))

        zs = {mode: [] for mode in modes}
        with torch.no_grad():
            for B in tqdm(dl):
                for mode in modes:
                    if mode == "sdss":
                        zs[mode].append(torch.tensor(np.array(B["embedding"])).T)
                    elif mode == "desi":
                        zs[mode].append(torch.tensor(np.array(B["embeddings"])).T)
                    else:
                        # Delegate embedding to the adapter implementation
                        outputs = adapter.embed_for_mode(B, mode)
                        zs[mode].append(outputs)


        zs = {mode: torch.cat(embs) for mode, embs in zs.items()}
        mknn_score = mknn(
            zs[modes[0]].cpu().numpy(), zs[modes[1]].cpu().numpy(), knn_k
        )

        print(f"\nmknn {model_alias}, {size}: {mknn_score:.8f}")

        # Create the directory if it doesn't exist
        os.makedirs("data", exist_ok=True)  
        # Creating the file to store mknn results
        with open(f"data/{comp_mode}_{model_alias}_mknn.txt", "a") as fi:
            fi.write(f"{size},{mknn_score:.8f}\n")

        df = df.with_columns(
            [
                pl.Series(
                    f"{model_alias}_{size.lstrip('0')}_{mode}".lower(),
                    embs.cpu().numpy(),
                )
                for mode, embs in zs.items()
            ]
        )

    df.write_parquet(f"data/{comp_mode}_{model_alias}.parquet")
    if upload_ds is not None:
        Dataset.from_polars(df).push_to_hub(upload_ds)
