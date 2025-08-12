

import os, pathlib, gc, shutil, warnings
import numpy as np
from PIL import Image, ImageFile
from datasets import load_dataset
import torch, timm, tqdm
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------- Config ----------
SEED = 42
MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", "10000"))  # set to 102000 for full run
BATCH_MKNN = 1000   # PRH commonly uses ~1000; set to None for full-set mKNN
REPEATS = 3         # averaged random batches for stability
Ks = [5, 10, 20, 50]

np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPUS = torch.cuda.device_count()
print(f"🟢 Device: {DEVICE} (GPUs: {N_GPUS})")

def clear_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()

# temp cache to avoid quota issues
TEMP_CACHE = pathlib.Path(f"/tmp/dino_prh_{os.getpid()}")
for sub in ("hub", "datasets", "models"): (TEMP_CACHE / sub).mkdir(parents=True, exist_ok=True)
os.environ.update({
    "HF_HOME": str(TEMP_CACHE),
    "HUGGINGFACE_HUB_CACHE": str(TEMP_CACHE / "hub"),
    "HF_DATASETS_CACHE": str(TEMP_CACHE / "datasets"),
    "TRANSFORMERS_CACHE": str(TEMP_CACHE / "models"),
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
})

# ---------- Data ----------
print("📥 Loading Legacy Survey/HSC cross-matched (streaming)…")
ds = load_dataset("Smith42/legacysurvey_hsc_crossmatched",
                  split="train", streaming=True,
                  cache_dir=str(TEMP_CACHE / "datasets"))

def flux_to_pil(blob):
    """Convert 2D/3D flux array to 3-channel PIL with per-image robust scaling."""
    arr = np.asarray(blob["flux"], np.float32)
    if arr.ndim == 3:  # just in case
        arr = arr[arr.shape[0] // 2]
    v0, v1 = np.nanpercentile(arr, 5), np.nanpercentile(arr, 99)
    if not np.isfinite(v0) or not np.isfinite(v1) or v1 <= v0: return None
    img = ((arr - v0) / (v1 - v0)).clip(0, 1) * 255
    img = img.astype(np.uint8)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    return Image.fromarray(img, "RGB")

Legacy_imgs, HSC_imgs = [], []
for i, ex in enumerate(tqdm.tqdm(ds, desc="Collecting pairs")):
    try:
        li = flux_to_pil(ex["legacysurvey_image"])
        hi = flux_to_pil(ex["hsc_image"])
        if li is not None and hi is not None:
            Legacy_imgs.append(li); HSC_imgs.append(hi)
    except Exception:
        pass
    if len(Legacy_imgs) >= MAX_SAMPLES:
        break

N = len(Legacy_imgs)
print(f"✅ Using {N} Legacy/HSC pairs (MAX_SAMPLES={MAX_SAMPLES})")
assert N >= 1000 or (BATCH_MKNN is None or BATCH_MKNN <= N), "Batch size for mKNN > N. Reduce BATCH_MKNN or increase MAX_SAMPLES."
clear_mem()

# ---------- Models: timm DINOv2 Base & Large ----------
def build_dino(name):
    print(f"🔍 Loading {name}…")
    net = timm.create_model(name, pretrained=True, num_classes=0,
                            cache_dir=str(TEMP_CACHE / "models"))
    net.eval().to(DEVICE)
    if N_GPUS > 1 and DEVICE.type == "cuda":
        net = torch.nn.DataParallel(net)
    return net

base_name, large_name = "vit_base_patch14_dinov2.lvd142m", "vit_large_patch14_dinov2.lvd142m"
base_net  = build_dino(base_name); clear_mem()
large_net = build_dino(large_name)

from timm.data import resolve_data_config, create_transform
def make_shared_eval_transform(ref_model):
    cfg = resolve_data_config({}, model=ref_model)
    cfg.update(dict(input_size=(3, 518, 518), interpolation="bicubic", crop_pct=1.0))
    return create_transform(**cfg)

shared_tf = make_shared_eval_transform(base_net)

# ---------- Embeddings (CLS pooled) ----------
@torch.no_grad()
def compute_embeddings(imgs, model, transform, tag="", bs=96):
    embs = []
    for i in tqdm.trange(0, len(imgs), bs, desc=f"Embed-{tag}"):
        batch = imgs[i:i+bs]
        x = torch.stack([transform(im) for im in batch]).to(DEVICE)
        with torch.cuda.amp.autocast(enabled=(DEVICE.type=="cuda")):
            z = model(x)  # num_classes=0 → pooled CLS feature
        embs.append(z.detach().float().cpu())
        del x, z; clear_mem()
    return torch.cat(embs, dim=0).numpy()

print("🔢 Computing embeddings…")
Legacy_B = compute_embeddings(Legacy_imgs, base_net,  shared_tf, "Legacy-Base", bs=96)
Legacy_L = compute_embeddings(Legacy_imgs, large_net, shared_tf, "Legacy-Large", bs=64)
HSC_B    = compute_embeddings(HSC_imgs,    base_net,  shared_tf, "HSC-Base",    bs=96)
HSC_L    = compute_embeddings(HSC_imgs,    large_net, shared_tf, "HSC-Large",   bs=64)

print("Shapes:",
      "\n Legacy Base ", Legacy_B.shape,
      "\n Legacy Large", Legacy_L.shape,
      "\n HSC    Base ", HSC_B.shape,
      "\n HSC    Large", HSC_L.shape)
assert Legacy_B.shape[1] != Legacy_L.shape[1], "Large & Base dims identical—did Large fail to load?"

# ---------- PRH mKNN (Appendix A, Eq. 11) ----------
def mknn_prh(Z1, Z2, k=10, batch_size=None, n_batches=1, seed=SEED):
    """Exact PRH Eq.11 mutual kNN overlap; exclude self; mean over i (and batches)."""
    assert Z1.shape[0] == Z2.shape[0], "Z1/Z2 must be row-aligned."
    N = Z1.shape[0]; rng = np.random.default_rng(seed)
    Z1 = Z1 / (np.linalg.norm(Z1, axis=1, keepdims=True) + 1e-8)
    Z2 = Z2 / (np.linalg.norm(Z2, axis=1, keepdims=True) + 1e-8)

    def one_batch(idxs):
        X1, X2 = Z1[idxs], Z2[idxs]
        b = len(idxs); kk = min(k, b-1)
        nn1 = NearestNeighbors(n_neighbors=kk+1, metric="euclidean").fit(X1)
        n1  = nn1.kneighbors(return_distance=False)[:, 1:]
        nn2 = NearestNeighbors(n_neighbors=kk+1, metric="euclidean").fit(X2)
        n2  = nn2.kneighbors(return_distance=False)[:, 1:]
        return np.mean([len(set(n1[i]) & set(n2[i])) / kk for i in range(b)])

    if batch_size is None or batch_size >= N:
        return float(one_batch(np.arange(N)))
    vals = []
    for t in range(n_batches):
        idxs = rng.choice(N, size=batch_size, replace=False)
        vals.append(one_batch(idxs))
    return float(np.mean(vals))

# ---------- Evaluate ----------
BATCH = min(BATCH_MKNN or N, N) if BATCH_MKNN is not None else None
print("\n📊 mKNN (PRH Eq.11) cross-modal Legacy ↔ HSC")
for k in Ks:
    base_cm = mknn_prh(Legacy_B, HSC_B, k=k, batch_size=BATCH, n_batches=REPEATS)
    large_cm= mknn_prh(Legacy_L, HSC_L, k=k, batch_size=BATCH, n_batches=REPEATS)
    print(f"  k={k:>2}  Base={base_cm:.4f}  Large={large_cm:.4f}  Δ(L−B)={large_cm-base_cm:+.4f}")

print("\n📊 mKNN (PRH Eq.11) same-modal Base ↔ Large")
for k in Ks:
    legacy_bvsl = mknn_prh(Legacy_B, Legacy_L, k=k, batch_size=BATCH, n_batches=REPEATS)
    hsc_bvsl    = mknn_prh(HSC_B,    HSC_L,    k=k, batch_size=BATCH, n_batches=REPEATS)
    print(f"  k={k:>2}  Legacy={legacy_bvsl:.4f}  HSC={hsc_bvsl:.4f}")

# ---------- Cleanup ----------
try:
    shutil.rmtree(TEMP_CACHE, ignore_errors=True)
    print(f"\n🧹 Cleaned cache: {TEMP_CACHE}")
except Exception:
    pass
