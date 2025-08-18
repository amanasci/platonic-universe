import os, pathlib, gc, shutil, warnings
import numpy as np
from PIL import Image, ImageFile
from datasets import load_dataset
import torch, tqdm
from sklearn.neighbors import NearestNeighbors
from transformers import ViTModel, ViTImageProcessor

warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPUS = torch.cuda.device_count()
print(f"🟢 Device: {DEVICE} (GPUs: {N_GPUS})")

MAX_SAMPLES   = int(os.getenv("MAX_SAMPLES", "10000"))  # set 102000 for full run
Ks            = (5, 10, 20, 50)
EVAL_SIZE     = 336   # works for both ViT-L (patch16) & ViT-H (patch14)
BATCH_IMG_L   = 24
BATCH_IMG_H   = 16
BATCH_MKNN    = 1000  # PRH uses ~1k; set None for full exact
REPEATS_MKNN  = 3

# Temp caches (avoid quota)
TEMP_CACHE = pathlib.Path(f"/tmp/vit_prh_{os.getpid()}")
for sub in ("hub", "datasets", "models"): (TEMP_CACHE / sub).mkdir(parents=True, exist_ok=True)
os.environ.update({
    "HF_HOME": str(TEMP_CACHE),
    "HUGGINGFACE_HUB_CACHE": str(TEMP_CACHE / "hub"),
    "HF_DATASETS_CACHE": str(TEMP_CACHE / "datasets"),
    "TRANSFORMERS_CACHE": str(TEMP_CACHE / "models"),
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
})

def clear_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()

# ────────────────────────────────────────────────────────────────────────
# Data: Legacy ↔ HSC (streaming)
# ────────────────────────────────────────────────────────────────────────
def flux_to_pil(blob, target=EVAL_SIZE):
    """Convert 2D/3D flux array to 3-ch PIL with robust per-image scaling."""
    arr = np.asarray(blob.get("flux", blob), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[arr.shape[0] // 2]
    if arr.size == 0:
        return None
    v0, v1 = np.nanpercentile(arr, 5), np.nanpercentile(arr, 99)
    if not np.isfinite(v0) or not np.isfinite(v1) or v1 <= v0:
        v0, v1 = float(np.nanmin(arr)), float(np.nanmax(arr))
        if not np.isfinite(v0) or not np.isfinite(v1) or v1 <= v0:
            return None
    img = ((arr - v0) / (v1 - v0)).clip(0, 1) * 255
    img = img.astype(np.uint8)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    im = Image.fromarray(img, "RGB")
    return im.resize((target, target), Image.BICUBIC)

print("📥 Loading Legacy Survey/HSC cross-matched (streaming)…")
ds = load_dataset("Smith42/legacysurvey_hsc_crossmatched",
                  split="train", streaming=True,
                  cache_dir=str(TEMP_CACHE / "datasets"))

Legacy_imgs, HSC_imgs = [], []
for ex in tqdm.tqdm(ds, desc="Collecting pairs"):
    try:
        li = ex.get("legacy_image", ex.get("legacysurvey_image", None))
        hi = ex.get("hsc_image", None)
        li = flux_to_pil(li) if li is not None else None
        hi = flux_to_pil(hi) if hi is not None else None
        if li is not None and hi is not None:
            Legacy_imgs.append(li); HSC_imgs.append(hi)
    except Exception:
        pass
    if len(Legacy_imgs) >= MAX_SAMPLES:
        break

N = len(Legacy_imgs)
print(f"✅ Using {N} pairs (MAX_SAMPLES={MAX_SAMPLES})")
assert N >= 1000 or BATCH_MKNN is None or BATCH_MKNN <= N, "Increase samples or reduce BATCH_MKNN."
clear_mem()

# ────────────────────────────────────────────────────────────────────────
# Models (Transformers ViT)
# ────────────────────────────────────────────────────────────────────────
def load_vit(name):
    print(f"🔍 Loading {name} …")
    proc = ViTImageProcessor.from_pretrained(name, cache_dir=str(TEMP_CACHE / "models"))
    mdl  = ViTModel.from_pretrained(name, cache_dir=str(TEMP_CACHE / "models"), use_safetensors=True)
    mdl  = mdl.to(DEVICE)
    if N_GPUS > 1 and DEVICE.type == "cuda":
        mdl = torch.nn.DataParallel(mdl)
    mdl.eval()
    return mdl, proc

vit_large_name = "google/vit-large-patch16-224"
vit_huge_name  = "google/vit-huge-patch14-224-in21k"

vit_large, proc_large = load_vit(vit_large_name); clear_mem()
vit_huge,  proc_huge  = load_vit(vit_huge_name)

# ────────────────────────────────────────────────────────────────────────
# Embedding extraction (patch-mean, ℓ2 norm) with pos-enc interpolation
# ────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def vit_embed(images, model, processor, tag="", bs=32, size=EVAL_SIZE, feat_mode="patch_mean"):
    """
    Returns (N, D) embeddings.
    - size: output H=W (we let processor resize to this)
    - feat_mode: 'patch_mean' (preferred) or 'cls'
    """
    chunks = []
    for s in tqdm.trange(0, len(images), bs, desc=f"Embed-{tag}"):
        batch = images[s:s+bs]
        inputs = processor(
            images=batch,
            return_tensors="pt",
            do_resize=True,
            size={"height": size, "width": size},
            do_center_crop=False,
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.inference_mode():
            try:
                out = model(**inputs, interpolate_pos_encoding=True)
            except TypeError:
                # very old transformers – fall back to model-native 224
                inputs_224 = processor(images=batch, return_tensors="pt")
                inputs_224 = {k: v.to(DEVICE) for k, v in inputs_224.items()}
                out = model(**inputs_224)

            h = out.last_hidden_state  # (B, 1+N, D)
            if feat_mode == "patch_mean" and h.size(1) > 1:
                feat = h[:, 1:, :].mean(dim=1)
            else:
                feat = h[:, 0, :]
            feat = torch.nn.functional.normalize(feat, dim=-1)
            chunks.append(feat.cpu().numpy())

        del inputs, out, h, feat
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.synchronize()
    return np.concatenate(chunks, 0)

print("🔢 Computing embeddings…")
Legacy_L = vit_embed(Legacy_imgs, vit_large, proc_large, "Legacy-L", bs=BATCH_IMG_L)
Legacy_H = vit_embed(Legacy_imgs, vit_huge,  proc_huge,  "Legacy-H", bs=BATCH_IMG_H)
HSC_L    = vit_embed(HSC_imgs,    vit_large, proc_large, "HSC-L",    bs=BATCH_IMG_L)
HSC_H    = vit_embed(HSC_imgs,    vit_huge,  proc_huge,  "HSC-H",    bs=BATCH_IMG_H)

print("Shapes:",
      "\n  Legacy L:", Legacy_L.shape,
      "\n  Legacy H:", Legacy_H.shape,
      "\n  HSC    L:", HSC_L.shape,
      "\n  HSC    H:", HSC_H.shape)

# ────────────────────────────────────────────────────────────────────────
# Exact PRH mKNN (App. A, Eq. 11)
# ────────────────────────────────────────────────────────────────────────
def mknn_prh(Z1, Z2, k=10, batch_size=None, repeats=1, seed=SEED):
    assert Z1.shape[0] == Z2.shape[0], "Row-aligned inputs required."
    N = Z1.shape[0]
    if N < 2: return 0.0

    # ℓ2 normalize (cosine == Euclid)
    Z1 = Z1 / (np.linalg.norm(Z1, axis=1, keepdims=True) + 1e-8)
    Z2 = Z2 / (np.linalg.norm(Z2, axis=1, keepdims=True) + 1e-8)
    rng = np.random.default_rng(seed)

    def one_batch(idxs):
        X1, X2 = Z1[idxs], Z2[idxs]
        b = len(idxs); kk = min(k, b - 1)
        nn1 = NearestNeighbors(n_neighbors=kk + 1, metric="euclidean").fit(X1)
        n1  = nn1.kneighbors(return_distance=False)[:, 1:]
        nn2 = NearestNeighbors(n_neighbors=kk + 1, metric="euclidean").fit(X2)
        n2  = nn2.kneighbors(return_distance=False)[:, 1:]
        return float(np.mean([len(set(n1[i]) & set(n2[i])) / kk for i in range(b)]))

    if batch_size is None or batch_size >= N:
        return one_batch(np.arange(N))
    vals = []
    for _ in range(repeats):
        idxs = rng.choice(N, size=batch_size, replace=False)
        vals.append(one_batch(idxs))
    return float(np.mean(vals))

# ────────────────────────────────────────────────────────────────────────
# Evaluate
# ────────────────────────────────────────────────────────────────────────
BATCH = min(BATCH_MKNN or N, N) if BATCH_MKNN is not None else None
print("\n📊 mKNN (PRH Eq.11) — Legacy ↔ HSC")
for k in Ks:
    large_cm = mknn_prh(Legacy_L, HSC_L, k=k, batch_size=BATCH, repeats=REPEATS_MKNN)
    huge_cm  = mknn_prh(Legacy_H, HSC_H, k=k, batch_size=BATCH, repeats=REPEATS_MKNN)
    print(f"  k={k:>2}  Large={large_cm:.4f}  Huge={huge_cm:.4f}  Δ(H−L)={huge_cm - large_cm:+.4f}")

print("\n📊 mKNN (PRH Eq.11) — Same-modal (Large ↔ Huge)")
for k in Ks:
    legacy_lh = mknn_prh(Legacy_L, Legacy_H, k=k, batch_size=BATCH, repeats=REPEATS_MKNN)
    hsc_lh    = mknn_prh(HSC_L,    HSC_H,    k=k, batch_size=BATCH, repeats=REPEATS_MKNN)
    print(f"  k={k:>2}  Legacy={legacy_lh:.4f}  HSC={hsc_lh:.4f}")

# ────────────────────────────────────────────────────────────────────────
# Cleanup
# ────────────────────────────────────────────────────────────────────────
try:
    shutil.rmtree(TEMP_CACHE, ignore_errors=True)
    print(f"\n🧹 Cleaned cache: {TEMP_CACHE}")
except Exception:
    pass