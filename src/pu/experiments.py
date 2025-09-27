import numpy as np
import polars as pl
import torch
from datasets import Dataset, load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
from astropt.model_utils import load_astropt

from pu.preprocess import PreprocessHF, PreprocessAstropt
from pu.metrics import mknn

def run_experiment(args):
    """Runs the embedding generation experiment based on the provided arguments."""
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
        sizes = ["small", "base", "large", "giant"]
        model_names = [f"facebook/dinov2-with-registers-{size}" for size in sizes]
    # ... (add other model definitions here)
    elif args.model == "astropt":
        sizes = ["015M", "095M", "850M"]
        model_names = [f"Smith42/astroPT_v2.0" for _ in sizes]
    else:
        raise NotImplementedError(f"Model '{args.model}' not implemented.")

    df = pl.DataFrame()
    for size, model_name in zip(sizes, model_names):
        if args.model == 'astropt':
            model = load_astropt(model_name, path=f"astropt/{size}").to("cuda")
            processor = PreprocessAstropt(model.modality_registry, modes, resize=False)
        else:
            model = AutoModel.from_pretrained(model_name).to("cuda")
            processor = PreprocessHF(modes, AutoImageProcessor.from_pretrained(model_name), resize=False)
        
        model.eval()

        # Dataset loading logic (remains the same as in original scripts)
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
                        if args.model == "astropt":
                            inputs = {
                                "images": B[f"{mode}_images"].to("cuda"),
                                "images_positions": B[f"{mode}_positions"].to("cuda"),
                            }
                            outputs = model.generate_embeddings(inputs)["images"].detach()
                        else:
                            inputs = B[f"{mode}"].to("cuda")
                            if args.model == "vit":
                                outputs = model(inputs).last_hidden_state[:, 1:].mean(dim=1).detach()
                            elif args.model == "convnext":
                                outputs = model(inputs).last_hidden_state.mean(dim=(2, 3)).detach()
                            elif args.model == "dino":
                                outputs = model(inputs).last_hidden_state[:, 0].detach()
                            elif args.model == "ijepa":
                                outputs = model(inputs).last_hidden_state.mean(dim=1).detach()
                            else:
                                raise NotImplementedError
                        zs[mode].append(outputs)


        zs = {mode: torch.cat(embs) for mode, embs in zs.items()}
        mknn_score = mknn(
            zs[modes[0]].cpu().numpy(), zs[modes[1]].cpu().numpy(), args.knn_k
        )

        print(f"\nmknn {args.model}, {size}: {mknn_score:.8f}")
        with open(f"data/{comp_mode}_{args.model}_mknn.txt", "a") as fi:
            fi.write(f"{size},{mknn_score:.8f}\n")

        df = df.with_columns(
            [
                pl.Series(
                    f"{args.model}_{size.lstrip('0')}_{mode}".lower(),
                    embs.cpu().numpy(),
                )
                for mode, embs in zs.items()
            ]
        )

    df.write_parquet(f"data/{comp_mode}_{args.model}.parquet")
    if upload_ds is not None:
        Dataset.from_polars(df).push_to_hub(upload_ds)