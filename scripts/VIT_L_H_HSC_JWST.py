import os, pathlib, shutil, gc, warnings
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch import nn
from transformers import ViTModel, ViTImageProcessor
from huggingface_hub import snapshot_download

from datasets import load_dataset
from sklearn.neighbors import NearestNeighbors
import tqdm

# ------------------------------ Config ---------------------------------
SEED          = 42
MAX_SAMPLES   = int(os.getenv("MAX_SAMPLES", "10000"))   # raise if you want full dataset
BATCH_BASE    = int(os.getenv("BATCH_BASE", "56"))       # use for Large (lighter)
BATCH_LARGE   = int(os.getenv("BATCH_LARGE", "40"))      # use for Huge (heavier)
Ks            = (5, 10, 20, 50)
PRINT_EVERY   = 2000

# Use -in21k checkpoints (standard, stronger transfer; same eval pipeline)
VIT_LARGE_ID  = os.getenv("VIT_LARGE_ID", "google/vit-large-patch16-224-in21k")
VIT_HUGE_ID   = os.getenv("VIT_HUGE_ID",  "google/vit-huge-patch14-224-in21k")

# Repro
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NGPU   = torch.cuda.device_count()
print(f"🟢 Device: {DEVICE} (GPUs: {NGPU})")

# ---------------------- Cache root selection (Quota-safe) --------------
def pick_cache_root(candidates=None, need_mb=1024):
    if candidates is None:
        candidates = ["/dev/shm", "/run/shm", "/scratch", "/local_scratch",
                      "/tmp", "/var/tmp", str(pathlib.Path.cwd() / "tmp")]
    print(f"🔍 Searching for writable cache location with {need_mb}MB free space...")
    for root in candidates:
        try:
            p = pathlib.Path(root)
            p.mkdir(parents=True, exist_ok=True)
            if not os.access(root, os.W_OK):
                print(f"   ❌ {root} - not writable"); continue
            stat = shutil.disk_usage(root)
            free_gb = stat.free / (1024**3)
            print(f"   📊 {root} - {free_gb:.2f}GB free", end="")
            if stat.free < need_mb * 1024**2:
                print(" - insufficient space"); continue
            # quick write test
            test = p / f"hf_cache_test_{os.getpid()}.tmp"
            with open(test, "wb") as f: f.write(b"ok")
            test.unlink(missing_ok=True)
            print(" - ✅ SELECTED")
            return p
        except Exception as e:
            print(f"   ❌ {root} - error: {e}")
            continue
    raise OSError("No suitable cache location found")

CACHE_ROOT = pick_cache_root()
TEMP_ROOT  = CACHE_ROOT / f"jwst_hsc_vit_{os.getpid()}"
for sub in ("dataset_local", "models", "dataset_cache"): (TEMP_ROOT / sub).mkdir(parents=True, exist_ok=True)

def clear_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# --------------------------- Data: HSC ↔ JWST only ---------------------
def materialize_dataset(repo_id="Smith42/jwst_hsc_crossmatched") -> pathlib.Path:
    """
    Download the dataset repo directly to a normal folder (no HF cache pointers).
    Returns the snapshot directory path.
    """
    print(f"📥 Downloading dataset snapshot: {repo_id}")
    snap_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(TEMP_ROOT / "dataset_local"),
        local_dir_use_symlinks=False,   # copy real files
        resume_download=True,
        max_workers=8,
    )
    print(f"✅ Snapshot at: {snap_dir}")
    return pathlib.Path(snap_dir)

def flux_to_pil(blob):
    """2D/3D 'flux' → 3-channel PIL; robust percentile scaling."""
    try:
        arr = np.asarray(blob["flux"], np.float32)
    except Exception:
        return None
    if arr.ndim == 3:
        arr = arr[arr.shape[0] // 2]  # middle band
    if arr.size == 0: return None
    v0, v1 = np.nanpercentile(arr, 5), np.nanpercentile(arr, 99)
    if not np.isfinite(v0) or not np.isfinite(v1) or v1 <= v0:
        v0, v1 = float(np.nanmin(arr)), float(np.nanmax(arr))
        if not np.isfinite(v0) or not np.isfinite(v1) or v1 <= v0:
            return None
    img = ((arr - v0) / (v1 - v0)).clip(0, 1)
    img = (img * 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, 2)
    return Image.fromarray(img, "RGB")

def load_hsc_jwst(max_samples=MAX_SAMPLES):
    """
    Load from the local snapshot path without touching HF cache;
    keep only the first MAX_SAMPLES valid pairs.
    """
    snap_dir = materialize_dataset("Smith42/jwst_hsc_crossmatched")

    print("🔗 Loading dataset from local snapshot …")
    from datasets import DownloadConfig
    download_config = DownloadConfig(
        cache_dir=str(TEMP_ROOT / "dataset_cache"),
    )
    try:
        ds = load_dataset(
            str(snap_dir), 
            split="train", 
            trust_remote_code=True,
            cache_dir=str(TEMP_ROOT / "dataset_cache"),
            download_config=download_config,
        )
    except Exception:
        ds = load_dataset(
            str(snap_dir), 
            split="train",
            cache_dir=str(TEMP_ROOT / "dataset_cache"),
            download_config=download_config,
        )

    HSC, JWST = [], []
    for i, ex in enumerate(tqdm.tqdm(ds, desc="Collecting pairs", unit="row")):
        try:
            hsc_blob  = ex.get("hsc_image",  ex.get("HSC_image", None))
            jwst_blob = ex.get("jwst_image", ex.get("JWST_image", None))
            if hsc_blob is None or jwst_blob is None:
                continue
            hi = flux_to_pil(hsc_blob)
            ji = flux_to_pil(jwst_blob)
            if hi is not None and ji is not None:
                HSC.append(hi); JWST.append(ji)
            if len(HSC) >= max_samples:
                break
            if i and i % PRINT_EVERY == 0:
                print(f"   …seen {i:,} rows, kept {len(HSC):,} pairs")
        except Exception:
            continue

    if len(HSC) < 100:
        raise RuntimeError(f"Too few pairs parsed ({len(HSC)}).")
    print(f"✅ Using {len(HSC):,} HSC/JWST pairs")
    return HSC, JWST

# ------------------------ Models & Preprocessing -----------------------
def build_vit(model_name):
    print(f"🔧 Loading {model_name} …")
    model = ViTModel.from_pretrained(
        model_name,
        cache_dir=str(TEMP_ROOT / "models"),
        use_safetensors=True,
        output_hidden_states=False,
    ).to(DEVICE)
    if NGPU > 1 and DEVICE.type == "cuda":
        model = nn.DataParallel(model)
    processor = ViTImageProcessor.from_pretrained(
        model_name,
        cache_dir=str(TEMP_ROOT / "models"),
    )
    model.eval()
    return model, processor

@torch.no_grad()
def compute_embeddings(images, model, processor, tag="", batch_size=48):
    """
    Standard ViT eval: processor at 224, center crop, CLS token only.
    Returns (N, D) float32 array (NOT normalized).
    """
    chunks = []
    for s in tqdm.trange(0, len(images), batch_size, desc=f"Embed-{tag}"):
        batch = images[s:s+batch_size]
        inputs = processor(images=batch, return_tensors="pt")
        px = inputs["pixel_values"].to(DEVICE)
        with torch.inference_mode():
            out = model(pixel_values=px)
            z = out.last_hidden_state[:, 0]  # CLS
            z = z.float().cpu().numpy()
        chunks.append(z)
        del inputs, px, out, z
        clear_mem()
    return np.concatenate(chunks, 0)

# -------------------------- PRH mKNN (Eq. 11) --------------------------
def mknn_prh(Z1, Z2, k=10, batch=None, repeats=3, seed=SEED):
    """
    Exact PRH Eq.11 mutual kNN overlap; cosine via L2-normalization then Euclid.
    """
    assert Z1.shape[0] == Z2.shape[0], "Row-aligned arrays required."
    N = Z1.shape[0]
    if N < 2: return 0.0

    def _norm(X):
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    A = _norm(Z1); B = _norm(Z2)
    rng = np.random.default_rng(seed)

    def one(idxs):
        X1, X2 = A[idxs], B[idxs]
        b = len(idxs); kk = min(k, b-1)
        nn1 = NearestNeighbors(n_neighbors=kk+1, metric="euclidean").fit(X1)
        n1  = nn1.kneighbors(return_distance=False)[:, 1:]
        nn2 = NearestNeighbors(n_neighbors=kk+1, metric="euclidean").fit(X2)
        n2  = nn2.kneighbors(return_distance=False)[:, 1:]
        return float(np.mean([len(set(n1[i]) & set(n2[i]))/kk for i in range(b)]))

    if batch is None or batch >= N:
        return one(np.arange(N))
    vals = []
    for _ in range(repeats):
        idxs = rng.choice(N, size=batch, replace=False)
        vals.append(one(idxs))
    return float(np.mean(vals))

# ------------------------------- Main ----------------------------------
def main():
    # 1) Data
    HSC_imgs, JWST_imgs = load_hsc_jwst(MAX_SAMPLES)

    # 2) Models (standard eval)
    print("🔍 Building ViT-L/16 and ViT-H/14 (in21k) with default processors…")
    vit_large, proc_large = build_vit(VIT_LARGE_ID)  # 1024-D
    vit_huge,  proc_huge  = build_vit(VIT_HUGE_ID)   # 1280-D

    # 3) Embeddings (CLS only, fp32)
    print("🔢 Computing embeddings… (same pipeline for both)")
    # Use the larger batch knob for Large (lighter) and the smaller for Huge.
    HSC_L = compute_embeddings(HSC_imgs,  vit_large, proc_large, "HSC-Large",  batch_size=BATCH_BASE)
    JWST_L= compute_embeddings(JWST_imgs, vit_large, proc_large, "JWST-Large", batch_size=BATCH_BASE)
    HSC_H = compute_embeddings(HSC_imgs,  vit_huge,  proc_huge,  "HSC-Huge",   batch_size=BATCH_LARGE)
    JWST_H= compute_embeddings(JWST_imgs, vit_huge,  proc_huge,  "JWST-Huge",  batch_size=BATCH_LARGE)

    # 4) Align (just in case)
    N = min(HSC_L.shape[0], JWST_L.shape[0], HSC_H.shape[0], JWST_H.shape[0])
    HSC_L, JWST_L = HSC_L[:N], JWST_L[:N]
    HSC_H, JWST_H = HSC_H[:N], JWST_H[:N]

    print("Shapes:")
    print("  HSC_L :", HSC_L.shape, " JWST_L:", JWST_L.shape)
    print("  HSC_H :", HSC_H.shape, " JWST_H:", JWST_H.shape)

    # 5) PRH mKNN
    BATCH_MKNN = min(1000, N)   # PRH often uses ~1k batches
    REPEATS    = 3

    print("\n📊 mKNN (PRH Eq.11) — HSC(ViT-Large) ↔ JWST(ViT-Large)")
    large_scores = {}
    for k in Ks:
        l = mknn_prh(HSC_L, JWST_L, k=k, batch=BATCH_MKNN, repeats=REPEATS)
        large_scores[k] = l
        print(f"  k={k:>2}  Large={l:.4f}")

    print("\n📊 mKNN (PRH Eq.11) — HSC(ViT-Huge) ↔ JWST(ViT-Huge)")
    huge_scores = {}
    for k in Ks:
        h = mknn_prh(HSC_H, JWST_H, k=k, batch=BATCH_MKNN, repeats=REPEATS)
        huge_scores[k] = h
        print(f"  k={k:>2}  Huge ={h:.4f}")

    print("\n📊 Δ(Huge − Large)")
    for k in Ks:
        print(f"  k={k:>2}  Δ={huge_scores[k]-large_scores[k]:+.4f}")

if __name__ == "__main__":
    try:
        main()
    finally:
        # Best-effort cleanup of temp area
        try:
            shutil.rmtree(TEMP_ROOT, ignore_errors=True)
        except Exception:
            pass
