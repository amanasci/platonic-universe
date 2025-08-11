import torch
import numpy as np

from datasets import load_dataset, Dataset

from astropt.model_utils import load_astropt
from astropt.local_datasets import GalaxyImageDataset

from functools import partial
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.neighbors import NearestNeighbors

from tqdm import tqdm
import polars as pl
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def data_transforms():
    return transforms.Compose([transforms.Lambda(normalise)])

def _process_galaxy_wrapper(idx, func, mode0, mode1):
    """This function ensures that the image is tokenised in the same way as the
    pre-trained model is expecting"""
    mode0_im = np.array(flux_to_pil(idx[f"{mode0}_image"])).swapaxes(0, 2)
    mode0_im = func(torch.from_numpy(mode0_im).to(torch.float)).to(torch.float)
    mode0_positions = torch.arange(0, len(mode0_im), dtype=torch.long)
    mode1_im = np.array(flux_to_pil(idx[f"{mode1}_image"])).swapaxes(0, 2)
    mode1_im = func(torch.from_numpy(mode1_im).to(torch.float)).to(torch.float)
    mode1_positions = torch.arange(0, len(mode1_im), dtype=torch.long)
    return {
        f"{mode0}_images": mode0_im,
        f"{mode0}_positions": mode0_positions,
        f"{mode1}_images": mode1_im,
        f"{mode1}_positions": mode1_positions,
    }

def flux_to_pil(blob):
    arr = np.asarray(blob["flux"], np.float32)
    if arr.ndim == 3: arr = arr[arr.shape[0] // 2]      # middle band
    v0, v1 = np.nanpercentile(arr, 5), np.nanpercentile(arr, 99)
    img = ((arr - v0) / (v1 - v0)).clip(0, 1) * 255
    img = img.astype(np.uint8)
    if img.ndim == 2: img = np.repeat(img[:, :, None], 3, axis=2)
    return Image.fromarray(img, "RGB")

mode0 = "hsc"
mode1 = "legacysurvey"
hf_ds = "Smith42/legacysurvey_hsc_crossmatched"
upload_ds = "UniverseTBD/legacysurvey_hsc_embeddings"

df = pl.DataFrame()

for size in ["015M", "095M", "850M"]:

    model = load_astropt("Smith42/astroPT_v2.0", path=f"astropt/{size}").to("cuda")
    
    galproc = GalaxyImageDataset(
      None,
      spiral=True,
      transform={"images": data_transforms()},
      modality_registry=model.modality_registry
    )
    
    ds = (
      load_dataset(hf_ds, split="train", streaming=True)
      .select_columns((f"{mode0}_image", f"{mode1}_image"))
      .map(partial(_process_galaxy_wrapper, func=galproc.process_galaxy, mode0=mode0, mode1=mode1))
      .remove_columns((f"{mode0}_image", f"{mode1}_image"))
    )
    
    dl = iter(DataLoader(ds, batch_size=128, num_workers=32))
    
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
    
    mknn_score = mknn(zs_mode0.cpu().numpy(), zs_mode1.cpu().numpy(), 10)
    
    print(f"\nMutual KNN Score: {mknn_score:.4f}")

    df = df.with_columns([
        pl.Series(f"astropt_{size.lstrip('0')}_{mode0}".lower(), zs_mode0.cpu().numpy()),
        pl.Series(f"astropt_{size.lstrip('0')}_{mode1}".lower(), zs_mode1.cpu().numpy())
    ])

print(df)
Dataset.from_polars(df).push_to_hub(upload_ds)
