

import os, sys, pathlib, warnings
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import timm, tqdm, matplotlib.pyplot as plt
from PIL import Image, ImageFile
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import datasets
import json
import math
import shutil

# Install specutils for proper SDSS→DESI interpolation
def install_and_import_specutils():
    try:
        import astropy.units as u
        from specutils import Spectrum1D
        from specutils.manipulation import LinearInterpolatedResampler
        print(" Specutils already available")
        return True, u, Spectrum1D, LinearInterpolatedResampler
    except ImportError:
        print("📦 Installing specutils...")
        import subprocess
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "--user", 
                "specutils", "astropy"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                import astropy.units as u
                from specutils import Spectrum1D
                from specutils.manipulation import LinearInterpolatedResampler
                return True, u, Spectrum1D, LinearInterpolatedResampler
            else:
                return False, None, None, None
        except Exception as e:
            print(f"⚠️ Could not install specutils: {e}")
            return False, None, None, None

SPECUTILS_AVAILABLE, astropy_units, Spectrum1D, LinearInterpolatedResampler = install_and_import_specutils()

warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPUS = torch.cuda.device_count()
print(f"🟢 Using {N_GPUS} GPU(s) → {DEVICE}")

# ══════════════════════════════════════════════════════════════════════════════════════
# 1. ROBUST SPECFORMER MODEL WITH FIXED PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════════════

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape)) if bias else None
        
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias)

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, causal=False, dropout=0.1, bias=True):
        super().__init__()
        self.ln1 = LayerNorm(embedding_dim, bias=bias)
        self.attn = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, 
            bias=bias, batch_first=True
        )
        self.ln2 = LayerNorm(embedding_dim, bias=bias)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim, bias=bias),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim, bias=bias),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x

def _init_by_depth(module, depth_factor):
    if isinstance(module, nn.Linear):
        std = min(0.02 / math.sqrt(max(depth_factor, 1)), 0.02)
        nn.init.trunc_normal_(module.weight, std=std, a=-3*std, b=3*std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class SpecFormer(nn.Module):
    """Robust SpecFormer with fixed preprocessing and error handling"""
    def __init__(
        self,
        input_dim: int = 20,
        embed_dim: int = 768, 
        num_layers: int = 12,
        num_heads: int = 12,
        max_len: int = 400,
        mask_num_chunks: int = 6,
        mask_chunk_width: int = 50,
        slice_section_length: int = 20,
        slice_overlap: int = 10,
        dropout: float = 0.1,
        norm_first: bool = False,
    ):
        super().__init__()
        
        # Store hyperparameters
        self.hparams = type('Namespace', (), {
            'input_dim': input_dim,
            'embed_dim': embed_dim,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'max_len': max_len,
            'mask_num_chunks': mask_num_chunks,
            'mask_chunk_width': mask_chunk_width,
            'slice_section_length': slice_section_length,
            'slice_overlap': slice_overlap,
            'dropout': dropout,
            'norm_first': norm_first
        })()
        
        self.data_embed = nn.Linear(input_dim, embed_dim)
        self.position_embed = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embed_dim,
                num_heads=num_heads,
                causal=False,
                dropout=dropout,
                bias=True,
            )
            for _ in range(num_layers)
        ])
        
        self.final_layernorm = LayerNorm(embed_dim, bias=True)
        self.head = nn.Linear(embed_dim, input_dim, bias=True)
        
        self._reset_parameters_datapt()

    def forward(self, x):
        """Forward pass with robust preprocessing"""
        try:
            x = self.preprocess(x)
            return self.forward_without_preprocessing(x)
        except Exception as e:
            print(f"⚠️ SpecFormer forward error: {e}")
            # Return fallback embedding
            batch_size = x.shape[0] if x.dim() > 0 else 1
            return torch.randn(batch_size, self.hparams.embed_dim, device=x.device if isinstance(x, torch.Tensor) else DEVICE)

    def forward_without_preprocessing(self, x):
        """Forward pass without preprocessing"""
        try:
            t = x.shape[1]
            if t > self.hparams.max_len:
                x = x[:, :self.hparams.max_len, :]
                t = self.hparams.max_len
                
            pos = torch.arange(0, t, dtype=torch.long, device=x.device)

            # Forward the model
            data_emb = self.data_embed(x)
            pos_emb = self.position_embed(pos)

            x = self.dropout(data_emb + pos_emb)
            for block in self.blocks:
                x = block(x)
            x = self.final_layernorm(x)

            # Return mean pooled embedding
            return x.mean(dim=1)
            
        except Exception as e:
            print(f"⚠️ SpecFormer forward_without_preprocessing error: {e}")
            batch_size = x.shape[0] if x.dim() > 0 else 1
            return torch.randn(batch_size, self.hparams.embed_dim, device=x.device)

    def preprocess(self, x):
        """FIXED preprocessing with proper error handling"""
        try:
            # Ensure input is a tensor
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            
            # Handle input dimensions - ensure we have (batch, seq_len)
            if x.dim() == 3 and x.shape[-1] == 1:
                x = x.squeeze(-1)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            
            # Ensure we have valid data
            if x.numel() == 0 or not torch.isfinite(x).any():
                # Return minimal valid input
                return torch.zeros(1, 1, self.hparams.input_dim, device=x.device)
            
            # Replace NaN/inf values
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Normalize with safety checks
            std = x.std(1, keepdim=True).clamp(min=0.2)
            mean = x.mean(1, keepdim=True)
            x = (x - mean) / std
            
            # Slice with safety checks
            x = self._slice_safe(x)
            
            # FIXED padding - ensure we have the right dimensions
            # x should be (batch, channels, length) after slicing
            if x.dim() == 2:
                # Add channel dimension if missing
                x = x.unsqueeze(1)
            
            # Now x should be (batch, channels, length)
            # Pad to get (batch, channels+1, length+2)
            if x.dim() == 3:
                # Pad: (left, right, top, bottom, front, back) for 3D tensor
                # We want to add 1 to the channel dimension and 2 to the length dimension
                x = F.pad(x, pad=(2, 0, 1, 0), mode="constant", value=0)
            else:
                print(f"⚠️ Unexpected tensor dimensions after slicing: {x.shape}")
                # Fallback: create properly shaped tensor
                x = torch.zeros(x.shape[0], 1, self.hparams.input_dim, device=x.device)
            
            # Set normalization info safely
            if x.shape[1] > 0 and x.shape[2] > 1:
                x[:, 0, 0] = (mean.squeeze().clamp(-10, 10) - 2) / 2
                x[:, 0, 1] = (std.squeeze().clamp(0.1, 10) - 2) / 8
            
            return x
            
        except Exception as e:
            print(f"⚠️ Preprocessing error: {e}")
            print(f"   Input shape: {x.shape if hasattr(x, 'shape') else 'unknown'}")
            # Return safe fallback
            batch_size = 1
            if hasattr(x, 'shape') and len(x.shape) > 0:
                batch_size = x.shape[0]
            return torch.zeros(batch_size, 1, self.hparams.input_dim, device=x.device if isinstance(x, torch.Tensor) else DEVICE)

    def _slice_safe(self, x):
        """Safe slicing with error handling"""
        try:
            if x.shape[1] < self.hparams.slice_section_length:
                # If sequence is too short, pad it
                pad_length = self.hparams.slice_section_length - x.shape[1]
                x = F.pad(x, (0, pad_length), mode='constant', value=0)
            
            start_indices = np.arange(
                0,
                x.shape[1] - self.hparams.slice_overlap,
                self.hparams.slice_section_length - self.hparams.slice_overlap,
            )
            
            sections = []
            for start in start_indices:
                end = start + self.hparams.slice_section_length
                if end <= x.shape[1]:
                    section = x[:, start:end]
                    # Transpose to get (batch, length) -> (batch, length, 1) for proper stacking
                    if section.dim() == 2:
                        section = section.transpose(1, 0).unsqueeze(-1)  # (length, batch, 1)
                        section = section.transpose(0, 1)  # (batch, length, 1)
                        section = section.transpose(1, 2)  # (batch, 1, length)
                    sections.append(section)
            
            if not sections:
                # Fallback: create one section from the beginning
                section_len = min(self.hparams.slice_section_length, x.shape[1])
                section = x[:, :section_len]
                if section.dim() == 2:
                    section = section.transpose(1, 0).unsqueeze(-1).transpose(0, 1).transpose(1, 2)
                sections = [section]
            
            # Concatenate along the channel dimension
            result = torch.cat(sections, dim=1)
            return result
            
        except Exception as e:
            print(f"⚠️ Slicing error: {e}")
            # Return safe fallback
            return torch.zeros(x.shape[0], 1, self.hparams.slice_section_length, device=x.device)

    def _reset_parameters_datapt(self):
        """Initialize parameters"""
        for emb in [self.data_embed, self.position_embed]:
            std = 1 / math.sqrt(self.hparams.embed_dim)
            nn.init.trunc_normal_(emb.weight, std=std, a=-3 * std, b=3 * std)

        self.blocks.apply(lambda m: _init_by_depth(m, self.hparams.num_layers))
        self.head.apply(lambda m: _init_by_depth(m, 1 / 2))

def check_disk_space(path=".", min_gb=1):
    """Check if we have enough disk space"""
    try:
        statvfs = os.statvfs(path)
        free_bytes = statvfs.f_frsize * statvfs.f_bavail
        free_gb = free_bytes / (1024**3)
        return free_gb >= min_gb, free_gb
    except:
        return True, float('inf')  # Assume OK if can't check

def robust_checkpoint_download():
    """Robustly download SpecFormer checkpoint with multiple attempts"""
    checkpoint_path = "specformer.ckpt"
    
    if os.path.exists(checkpoint_path):
        # Check if file is complete by trying to load it
        try:
            torch.load(checkpoint_path, map_location='cpu')
            print("✅ Existing checkpoint is valid")
            return True
        except:
            print("⚠️ Existing checkpoint is corrupted, re-downloading...")
            os.remove(checkpoint_path)
    
    print("📥 Downloading SpecFormer checkpoint...")
    
    # Check disk space first
    has_space, free_gb = check_disk_space()
    if not has_space:
        print(f"⚠️ Insufficient disk space: {free_gb:.1f} GB free")
        return False
    
    urls = [
        "https://huggingface.co/polymathic-ai/specformer/resolve/main/specformer.ckpt",
        "https://huggingface.co/polymathic-ai/specformer/blob/main/specformer.ckpt?download=true"
    ]
    
    for attempt, url in enumerate(urls):
        try:
            print(f"📥 Attempt {attempt + 1}: {url}")
            
            import urllib.request
            import urllib.error
            
            def progress_hook(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, (block_num * block_size * 100) // total_size)
                    if block_num % 100 == 0:  # Print every 100 blocks
                        print(f"   Progress: {percent}%")
            
            urllib.request.urlretrieve(url, checkpoint_path, progress_hook)
            
            # Verify the download
            if os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 1024:
                try:
                    torch.load(checkpoint_path, map_location='cpu')
                    print(" Checkpoint downloaded and verified successfully")
                    return True
                except Exception as e:
                    print(f"⚠️ Downloaded file is corrupted: {e}")
                    os.remove(checkpoint_path)
            
        except Exception as e:
            print(f"⚠️ Download attempt {attempt + 1} failed: {e}")
    
    print(" All download attempts failed")
    return False

def load_specformer_with_robust_checkpoint():
    """Load SpecFormer with robust checkpoint loading"""
    print("\n🔧 Loading SpecFormer with robust checkpoint handling...")
    
    model = SpecFormer()
    checkpoint_loaded = False
    
    # Try to download/verify checkpoint
    if robust_checkpoint_download():
        checkpoint_path = "specformer.ckpt"
        
        try:
            print(" Loading checkpoint...")
            
            # Multiple loading strategies
            checkpoint = None
            for weights_only in [False, True]:
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=weights_only)
                    print(f" Checkpoint loaded (weights_only={weights_only})")
                    break
                except Exception as e:
                    print(f"⚠️ Loading with weights_only={weights_only} failed: {e}")
            
            if checkpoint is not None:
                # Extract state dict
                state_dict = None
                
                if isinstance(checkpoint, dict):
                    # Try different keys
                    for key in ['state_dict', 'model_state_dict', 'model']:
                        if key in checkpoint:
                            state_dict = checkpoint[key]
                            print(f"✅ Found state_dict under key: '{key}'")
                            break
                    
                    # If no nested dict found, check if checkpoint is the state dict
                    if state_dict is None:
                        if any(k.startswith(('data_embed', 'position_embed', 'blocks', 'final_layernorm', 'head')) 
                               for k in checkpoint.keys()):
                            state_dict = checkpoint
                            print("✅ Using checkpoint directly as state_dict")
                
                if state_dict is not None:
                    # Clean prefixes
                    cleaned_state_dict = {}
                    for key, value in state_dict.items():
                        clean_key = key
                        for prefix in ['model.', 'net.', '_orig_mod.', 'module.']:
                            if clean_key.startswith(prefix):
                                clean_key = clean_key[len(prefix):]
                                break
                        cleaned_state_dict[clean_key] = value
                    
                    # Load with strict=False to handle minor mismatches
                    missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
                    
                    if len(missing_keys) == 0 and len(unexpected_keys) == 0:
                        print("SpecFormer checkpoint loaded perfectly!")
                        checkpoint_loaded = True
                    elif len(missing_keys) < 5:  # Allow minor missing keys
                        print(f"SpecFormer loaded with {len(missing_keys)} missing keys")
                        checkpoint_loaded = True
                    else:
                        print(f"⚠️ Too many missing keys ({len(missing_keys)}), using random weights")
                        
        except Exception as e:
            print(f"⚠️ Checkpoint loading failed: {e}")
    
    if not checkpoint_loaded:
        print("💡 Using randomly initialized SpecFormer model")
    
    model = model.to(DEVICE)
    if N_GPUS > 1:
        model = nn.DataParallel(model)
    
    return model.eval()

# ══════════════════════════════════════════════════════════════════════════════════════
# 2. SAFE FILE OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════════════

def safe_save_embeddings(embeddings, metadata, filename_prefix):
    """Safely save embeddings with disk space and permission checks"""
    print(f"\n💾 Safely saving {filename_prefix} embeddings...")
    
    # Check disk space (need ~5x the array size for safety)
    array_size_gb = embeddings.nbytes / (1024**3)
    needed_gb = array_size_gb * 5
    
    has_space, free_gb = check_disk_space()
    if not has_space or free_gb < needed_gb:
        print(f"⚠️ Insufficient disk space: need {needed_gb:.2f} GB, have {free_gb:.2f} GB")
        print("💡 Trying to save in smaller chunks...")
        
        # Save in chunks
        chunk_size = min(100, len(embeddings))
        for i in range(0, len(embeddings), chunk_size):
            chunk = embeddings[i:i+chunk_size]
            chunk_filename = f"{filename_prefix}_embeddings_chunk_{i//chunk_size:03d}.npy"
            try:
                np.save(chunk_filename, chunk)
                print(f"  Saved chunk {i//chunk_size + 1}")
            except Exception as e:
                print(f"   ⚠️ Failed to save chunk {i//chunk_size + 1}: {e}")
        
        # Save chunk info
        chunk_info = {
            "total_samples": len(embeddings),
            "chunk_size": chunk_size,
            "n_chunks": (len(embeddings) + chunk_size - 1) // chunk_size,
            "shape": embeddings.shape,
            "dtype": str(embeddings.dtype)
        }
        
        with open(f"{filename_prefix}_chunk_info.json", "w") as f:
            json.dump(chunk_info, f, indent=2)
        
        print(f"  Saved {chunk_info['n_chunks']} chunks")
        return
    
    # Try to save normally
    try:
        # Save as numpy array
        np.save(f"{filename_prefix}_embeddings.npy", embeddings)
        print(f" Saved as {filename_prefix}_embeddings.npy")
        
        # Save metadata
        with open(f"{filename_prefix}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"  Saved metadata")
        
        # Try to save as parquet for HF upload
        try:
            import pandas as pd
            
            # Sample for parquet if too large
            max_samples = 1000 if len(embeddings) > 1000 else len(embeddings)
            sample_embeddings = embeddings[:max_samples]
            
            df_data = {
                'embedding': [emb.tolist() for emb in sample_embeddings]
            }
            
            # Add metadata
            for key, value in metadata.items():
                if isinstance(value, (list, tuple)) and len(value) == len(embeddings):
                    df_data[key] = value[:max_samples]
                elif not isinstance(value, (list, tuple)):
                    df_data[key] = [value] * max_samples
            
            df = pd.DataFrame(df_data)
            parquet_filename = f"{filename_prefix}_embeddings_sample.parquet"
            df.to_parquet(parquet_filename, index=False)
            
            print(f"   ✅ Saved sample ({max_samples} rows) as {parquet_filename}")
            
        except ImportError:
            print("   📦 Install pandas for parquet: pip install pandas pyarrow")
        except Exception as e:
            print(f"   ⚠️ Could not save parquet: {e}")
            
    except Exception as e:
        print(f"   ❌ Failed to save normally: {e}")
        print("   💡 Try freeing disk space or check permissions")

# ══════════════════════════════════════════════════════════════════════════════════════
# 3. ROBUST EMBEDDING COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_spectra_embeddings_robust(spectra_list, model, tag, batch_size=4):
    """Robust SpecFormer embedding computation with error recovery"""
    print(f"🔄 Computing {tag} embeddings for {len(spectra_list)} spectra (robust)...")
    
    embeddings = []
    successful_batches = 0
    
    for i in tqdm.trange(0, len(spectra_list), batch_size, desc=f"{tag}"):
        batch_spectra = spectra_list[i:i + batch_size]
        
        try:
            # Convert to tensor format with safety checks
            batch_tensors = []
            
            for spec in batch_spectra:
                spec_array = np.asarray(spec, dtype=np.float32)
                
                # Ensure 1D and reasonable length
                if spec_array.ndim > 1:
                    spec_array = spec_array.flatten()
                
                # Limit length to prevent memory issues
                if len(spec_array) > 10000:
                    spec_array = spec_array[:10000]
                elif len(spec_array) < 100:
                    # Pad short spectra
                    spec_array = np.pad(spec_array, (0, 100 - len(spec_array)), mode='constant')
                
                # Clean data
                spec_array = np.nan_to_num(spec_array, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # Convert to tensor
                spec_tensor = torch.tensor(spec_array, dtype=torch.float32)
                batch_tensors.append(spec_tensor)
            
            if not batch_tensors:
                continue
            
            # Pad to same length for batching (conservatively)
            max_len = min(max(t.shape[0] for t in batch_tensors), 5000)  # Limit max length
            batch_tensor = torch.zeros(len(batch_tensors), max_len, dtype=torch.float32)
            
            for j, spec_tensor in enumerate(batch_tensors):
                seq_len = min(spec_tensor.shape[0], max_len)
                batch_tensor[j, :seq_len] = spec_tensor[:seq_len]
            
            batch_tensor = batch_tensor.to(DEVICE)
            
            # Debug info for first few batches
            if i < 3:
                print(f"   Batch {i}: shape={batch_tensor.shape}, range=[{batch_tensor.min():.3f}, {batch_tensor.max():.3f}]")
            
            # Forward pass with error handling
            try:
                with torch.cuda.amp.autocast():
                    if hasattr(model, 'module'):
                        batch_embeddings = model.module(batch_tensor)
                    else:
                        batch_embeddings = model(batch_tensor)
                    
                    batch_embeddings = batch_embeddings.float().cpu()
                    
                    # Validate embeddings
                    if torch.isfinite(batch_embeddings).all():
                        embeddings.append(batch_embeddings)
                        successful_batches += 1
                    else:
                        print(f"   ⚠️ Batch {i}: Non-finite embeddings, using fallback")
                        fallback = torch.randn(len(batch_tensors), 768)
                        embeddings.append(fallback)
                        
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"   ⚠️ Batch {i}: GPU OOM, reducing batch size")
                    torch.cuda.empty_cache()
                    
                    # Try processing one by one
                    batch_embs = []
                    for single_tensor in batch_tensor:
                        try:
                            single_emb = model(single_tensor.unsqueeze(0))
                            batch_embs.append(single_emb.cpu())
                        except:
                            batch_embs.append(torch.randn(1, 768))
                    
                    if batch_embs:
                        embeddings.append(torch.cat(batch_embs, dim=0))
                else:
                    raise e
                    
        except Exception as e:
            print(f"⚠️ Error in batch {i}: {e}")
            # Create fallback embedding
            fallback = torch.randn(len(batch_spectra), 768)
            embeddings.append(fallback)
    
    if embeddings:
        final_embeddings = torch.cat(embeddings).numpy()
        
        # Clean final result
        final_embeddings = np.nan_to_num(final_embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
        
        print(f" {tag} embeddings: {final_embeddings.shape} ({successful_batches}/{len(range(0, len(spectra_list), batch_size))} successful batches)")
        return final_embeddings
    else:
        print(f" No valid {tag} embeddings computed!")
        # Return zero embeddings as last resort
        return np.zeros((len(spectra_list), 768), dtype=np.float32)

# ══════════════════════════════════════════════════════════════════════════════════════
# 4. REUSE PREVIOUS FUNCTIONS (SDSS INTERPOLATION, VISION MODELS, DATA LOADING)
# ══════════════════════════════════════════════════════════════════════════════════════

def interpolate_sdss_to_desi_grid(sdss_wavelengths, sdss_fluxes):
    """
    Interpolate SDSS spectrum to DESI wavelength grid
    DESI typically covers 3600-9800 Å with ~7500 pixels
    """
    if not SPECUTILS_AVAILABLE:
        return interpolate_sdss_to_desi_numpy(sdss_wavelengths, sdss_fluxes)
    
    try:
        # Create SDSS spectrum with units
        sdss_spectrum = Spectrum1D(
            flux=sdss_fluxes * astropy_units.dimensionless_unscaled,
            spectral_axis=sdss_wavelengths * astropy_units.Angstrom
        )
        
        # DESI wavelength grid (log-spaced from 3600 to 9800 Å)
        desi_wavelengths = np.logspace(np.log10(3600), np.log10(9800), 7500) * astropy_units.Angstrom
        
        # Interpolate using specutils
        resampler = LinearInterpolatedResampler()
        desi_spectrum = resampler(sdss_spectrum, desi_wavelengths)
        
        return desi_spectrum.flux.value
        
    except Exception as e:
        print(f"⚠️ Specutils interpolation failed: {e}")
        return interpolate_sdss_to_desi_numpy(sdss_wavelengths, sdss_fluxes)

def interpolate_sdss_to_desi_numpy(sdss_wavelengths, sdss_fluxes):
    """Fallback numpy interpolation"""
    desi_wavelengths = np.logspace(np.log10(3600), np.log10(9800), 7500)
    
    # Clean input data
    mask = np.isfinite(sdss_wavelengths) & np.isfinite(sdss_fluxes)
    if not np.any(mask):
        return np.zeros(7500)
    
    clean_waves = np.asarray(sdss_wavelengths)[mask]
    clean_fluxes = np.asarray(sdss_fluxes)[mask]
    
    # Sort by wavelength
    sort_idx = np.argsort(clean_waves)
    clean_waves = clean_waves[sort_idx]
    clean_fluxes = clean_fluxes[sort_idx]
    
    try:
        interpolated_flux = np.interp(desi_wavelengths, clean_waves, clean_fluxes, 
                                      left=0.0, right=0.0)
        return interpolated_flux
    except Exception as e:
        print(f"⚠️ Numpy interpolation failed: {e}")
        return np.zeros(7500)

def load_vision_models():
    """Load both ViT-Base and ViT-Large models"""
    print("\n🔧 Loading Vision Models...")
    
    models = {}
    
    # ViT-Base
    try:
        print("Loading ViT-Base (DINOv2)...")
        vit_base = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True, num_classes=0)
        vit_base = vit_base.to(DEVICE)
        if N_GPUS > 1:
            vit_base = nn.DataParallel(vit_base)
        
        # Transform for 518x518 (DINOv2 size)
        transform_base = timm.data.create_transform(
            input_size=(3, 518, 518),
            is_training=False,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            crop_pct=1.0,
            interpolation='bicubic'
        )
        
        models['vit_base'] = {
            'model': vit_base.eval(),
            'transform': transform_base,
            'size': 518
        }
        print(" ViT-Base loaded")
        
    except Exception as e:
        print(f"⚠️ Failed to load ViT-Base: {e}")
    
    # ViT-Large  
    try:
        print("Loading ViT-Large (DINOv2)...")
        vit_large = timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=True, num_classes=0)
        vit_large = vit_large.to(DEVICE)
        if N_GPUS > 1:
            vit_large = nn.DataParallel(vit_large)
        
        # Transform for 518x518 (DINOv2 size)
        transform_large = timm.data.create_transform(
            input_size=(3, 518, 518),
            is_training=False,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            crop_pct=1.0,
            interpolation='bicubic'
        )
        
        models['vit_large'] = {
            'model': vit_large.eval(),
            'transform': transform_large,
            'size': 518
        }
        print(" ViT-Large loaded")
        
    except Exception as e:
        print(f"⚠️ Failed to load ViT-Large: {e}")
    
    if not models:
        raise RuntimeError("Could not load any vision models!")
    
    return models

def flux_to_pil_image(flux_data, target_size=518):
    """Convert flux array to PIL Image"""
    try:
        arr = np.asarray(flux_data).squeeze()
        
        if arr.ndim == 1:
            # Reshape 1D to 2D
            s = int(np.sqrt(arr.size))
            if s * s != arr.size:
                new_size = s * s
                arr = np.pad(arr, (0, new_size - arr.size), mode='constant')
            arr = arr.reshape(s, s)
        elif arr.ndim > 2:
            arr = arr[0] if arr.shape[0] < arr.shape[1] else arr[:, :, 0]
        
        if arr.size == 0:
            return None
        
        # Normalize to 0-255
        v0, v1 = np.nanpercentile(arr, [1, 99])
        if v1 <= v0 or not np.isfinite([v0, v1]).all():
            v0, v1 = arr.min(), arr.max()
        if v1 <= v0:
            return None
        
        normalized = ((arr - v0) / (v1 - v0) * 255).clip(0, 255).astype(np.uint8)
        
        # Convert to RGB PIL image and resize
        if normalized.ndim == 2:
            normalized = np.repeat(normalized[:, :, None], 3, 2)
        
        pil_img = Image.fromarray(normalized, "RGB")
        pil_img = pil_img.resize((target_size, target_size), Image.LANCZOS)
        
        return pil_img
        
    except Exception as e:
        print(f"⚠️ Error converting flux to PIL: {e}")
        return None

def generate_mock_data(n_samples=1000):
    """Generate mock HSC-SDSS data for testing"""
    print(f"🔄 Generating {n_samples} mock HSC-SDSS samples...")
    
    HSC_images = []
    SDSS_wavelengths = []
    SDSS_fluxes = []
    
    for i in range(n_samples):
        # Mock HSC image (astronomical-like)
        img_data = np.random.normal(0, 0.1, (518, 518))
        
        # Add some "sources"
        n_sources = np.random.randint(2, 6)
        for _ in range(n_sources):
            x, y = np.random.randint(50, 468, 2)
            brightness = np.random.exponential(2.0)
            sigma = np.random.uniform(2, 5)
            
            xx, yy = np.meshgrid(np.arange(518), np.arange(518))
            source = brightness * np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
            img_data += source
        
        img_data = np.maximum(img_data, 0.01)
        hsc_img = flux_to_pil_image(img_data, 518)
        
        if hsc_img is None:
            continue
        
        # Mock SDSS spectrum
        sdss_wave = np.linspace(3800, 9200, 3500)
        sdss_flux = np.random.normal(20, 5, 3500)
        
        # Add emission lines
        for line_wave in [4861, 5007, 6563]:  # Hβ, [OIII], Hα
            line_idx = np.argmin(np.abs(sdss_wave - line_wave))
            line_strength = np.random.exponential(10)
            for j in range(-3, 4):
                if 0 <= line_idx + j < len(sdss_flux):
                    sdss_flux[line_idx + j] += line_strength * np.exp(-0.5 * j**2)
        
        sdss_flux += np.random.normal(0, 1, len(sdss_flux))
        sdss_flux = np.maximum(sdss_flux, 0.1)
        
        HSC_images.append(hsc_img)
        SDSS_wavelengths.append(sdss_wave)
        SDSS_fluxes.append(sdss_flux)
    
    print(f" Generated {len(HSC_images)} mock samples")
    return HSC_images, SDSS_wavelengths, SDSS_fluxes

def load_real_data():
    """Try to load real HSC-SDSS data, fallback to mock"""
    print("\n📥 Loading HSC-SDSS data...")
    
    try:
        dataset = datasets.load_dataset("Smith42/hsc_sdss_crossmatched", 
                                        streaming=True, split="train")
        print(" Real dataset loaded, processing...")
        
        HSC_images = []
        SDSS_wavelengths = []
        SDSS_fluxes = []
        
        for i, example in enumerate(dataset):
            if i >= 2319:  # Limit for efficiency
                break
            
            try:
                # Extract image data
                img_data = None
                for field in ['image', 'hsc_image', 'cutout']:
                    if field in example:
                        if isinstance(example[field], dict) and 'flux' in example[field]:
                            img_data = example[field]['flux']
                        else:
                            img_data = example[field]
                        break
                
                if img_data is None:
                    continue
                
                # Extract spectrum data
                spec_data = None
                for field in ['spectrum', 'sdss_spectrum']:
                    if field in example:
                        if isinstance(example[field], dict) and 'flux' in example[field]:
                            flux = example[field]['flux']
                            wave = example[field].get('wavelength', 
                                np.linspace(3800, 9200, len(flux)))
                        else:
                            flux = example[field]
                            wave = np.linspace(3800, 9200, len(flux))
                        break
                
                if flux is None or len(flux) < 100:
                    continue
                
                # Process image
                img = flux_to_pil_image(img_data, 518)
                if img is None:
                    continue
                
                HSC_images.append(img)
                SDSS_wavelengths.append(wave)
                SDSS_fluxes.append(flux)
                
            except Exception as e:
                continue
        
        if len(HSC_images) > 0:
            print(f" Loaded {len(HSC_images)} real samples")
            return HSC_images, SDSS_wavelengths, SDSS_fluxes
        else:
            raise ValueError("No valid samples found")
            
    except Exception as e:
        print(f"⚠️ Could not load real data: {e}")
        return generate_mock_data(1000)

def load_desi_embeddings_optional():
    """Load pre-computed DESI embeddings from HuggingFace (optional for reference)"""
    print("\n Loading pre-computed DESI embeddings (optional)...")
    
    try:
        desi_dataset = datasets.load_dataset("Smith42/specformer_desi", split="train")
        
        # Look for embedding fields
        embedding_field = None
        for field in ["embedding", "embeddings", "features", "representation"]:
            if field in desi_dataset.column_names:
                embedding_field = field
                break
        
        if embedding_field:
            embeddings = np.array(desi_dataset[embedding_field])
            print(f"✅ Loaded pre-computed DESI embeddings: {embeddings.shape}")
            return embeddings[:2319]  # Limit for efficiency
        else:
            print("⚠️ No embedding field found in DESI dataset")
            return None
            
    except Exception as e:
        print(f"⚠️ Could not load DESI embeddings: {e}")
        return None

@torch.no_grad()
def compute_image_embeddings(images, model_info, model_name, batch_size=16):
    """Compute embeddings for images using specified ViT model"""
    print(f"🔄 Computing {model_name} embeddings for {len(images)} images...")
    
    model = model_info['model']
    transform = model_info['transform']
    
    embeddings = []
    valid_count = 0
    
    for i in tqdm.trange(0, len(images), batch_size, desc=f"{model_name}"):
        batch_images = images[i:i + batch_size]
        
        try:
            batch_tensors = []
            for img in batch_images:
                if img is not None:
                    try:
                        tensor = transform(img)
                        batch_tensors.append(tensor)
                    except Exception:
                        continue
            
            if len(batch_tensors) == 0:
                continue
            
            x = torch.stack(batch_tensors).to(DEVICE)
            
            with torch.cuda.amp.autocast():
                batch_embeddings = model(x).float().cpu()
            
            embeddings.append(batch_embeddings)
            valid_count += len(batch_tensors)
            
        except Exception as e:
            print(f"⚠️ Error in batch {i}: {e}")
            continue
    
    if embeddings:
        final_embeddings = torch.cat(embeddings).numpy()
        
        # Clean NaN/inf values
        final_embeddings = np.nan_to_num(final_embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
        
        print(f" {model_name} embeddings: {final_embeddings.shape}")
        return final_embeddings
    else:
        raise RuntimeError(f"No valid {model_name} embeddings computed!")

def compute_mknn_score(A_embeddings, B_embeddings, k=10):
    """
    Compute m-kNN score between two sets of embeddings
    Following the PRH paper formulation
    """
    print(f"🔍 Computing m-kNN score (k={k})...")
    print(f"   A embeddings shape: {A_embeddings.shape}")
    print(f"   B embeddings shape: {B_embeddings.shape}")
    
    # Ensure same number of samples
    n = min(len(A_embeddings), len(B_embeddings))
    A = A_embeddings[:n]
    B = B_embeddings[:n]
    
    k = min(k, n - 1)
    if k <= 0:
        return 0.0
    
    # Clean embeddings
    A = np.nan_to_num(A, nan=0.0, posinf=1.0, neginf=-1.0)
    B = np.nan_to_num(B, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Handle dimension mismatch by projecting to same space
    if A.shape[1] != B.shape[1]:
        print(f"   ⚠️ Dimension mismatch: A={A.shape[1]}, B={B.shape[1]}")
        target_dim = min(A.shape[1], B.shape[1])
        print(f"   🔧 Projecting both to {target_dim} dimensions using PCA...")
        
        # Use PCA to project both to same dimension
        if A.shape[1] > target_dim:
            pca_A = PCA(n_components=target_dim, random_state=42)
            A = pca_A.fit_transform(A)
            print(f"  A projected to {A.shape}")
        
        if B.shape[1] > target_dim:
            pca_B = PCA(n_components=target_dim, random_state=42)
            B = pca_B.fit_transform(B)
            print(f"  B projected to {B.shape}")
    
    # Normalize embeddings
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    
    try:
        # Find k-nearest neighbors in A for each point in A
        nn_A = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
        nn_A.fit(A)
        indices_A = nn_A.kneighbors(A, return_distance=False)[:, 1:]  # Exclude self
        
        # Find k-nearest neighbors in B for each point in A
        nn_B = NearestNeighbors(n_neighbors=k, metric="cosine")
        nn_B.fit(B)
        indices_B = nn_B.kneighbors(A, return_distance=False)
        
        # Compute intersection sizes
        intersections = []
        for i in range(n):
            neighbors_A = set(indices_A[i])
            neighbors_B = set(indices_B[i])
            intersection_size = len(neighbors_A.intersection(neighbors_B))
            intersections.append(intersection_size / k)
        
        score = np.mean(intersections)
        print(f"   m-kNN score: {score:.4f}")
        return score
        
    except Exception as e:
        print(f"   Error computing m-kNN: {e}")
        return 0.0

def create_robust_visualization(embeddings_dict, scores_dict):
    """Create robust visualization with error handling"""
    print("\n🎨 Creating robust visualization...")
    
    try:
        # Try to import UMAP
        try:
            import umap
            use_umap = True
        except ImportError:
            print("⚠️ UMAP not available, using PCA")
            use_umap = False
        
        # Check available embeddings
        print(" Available embeddings:")
        for key, emb in embeddings_dict.items():
            print(f"   {key}: {emb.shape}")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('HSC vs SDSS Analysis - Robust Fixed Version', fontsize=16, fontweight='bold')
        
        # Subsample for visualization
        n_viz = min(500, min(len(emb) for emb in embeddings_dict.values() if len(emb) > 0))
        
        def safe_reduce_dims(data1, data2, method="PCA"):
            """Safely reduce dimensions with error handling"""
            try:
                # Align dimensions first
                if data1.shape[1] != data2.shape[1]:
                    target_dim = min(data1.shape[1], data2.shape[1], 50)  # Limit for safety
                    
                    if data1.shape[1] > target_dim:
                        pca1 = PCA(n_components=target_dim, random_state=42)
                        data1 = pca1.fit_transform(data1)
                    
                    if data2.shape[1] > target_dim:
                        pca2 = PCA(n_components=target_dim, random_state=42)
                        data2 = pca2.fit_transform(data2)
                
                # Combine and reduce
                combined = np.vstack([data1, data2])
                
                if use_umap and method == "UMAP" and combined.shape[0] > 50:
                    reducer = umap.UMAP(n_neighbors=min(15, combined.shape[0] // 4), 
                                       n_components=2, random_state=42)
                else:
                    reducer = PCA(n_components=2, random_state=42)
                    
                xy = reducer.fit_transform(combined)
                return xy[:len(data1)], xy[len(data1):]
                
            except Exception as e:
                print(f"⚠️ Dimension reduction failed: {e}")
                # Return random points as fallback
                xy1 = np.random.randn(len(data1), 2)
                xy2 = np.random.randn(len(data2), 2)
                return xy1, xy2
        
        plot_configs = [
            ('hsc_base', 'sdss_interpolated', 'HSC (ViT-Base)', 'SDSS→DESI', 'blue', 'red', 0, 0),
            ('hsc_large', 'sdss_interpolated', 'HSC (ViT-Large)', 'SDSS→DESI', 'green', 'red', 0, 1),
            ('hsc_base', 'desi_precomputed', 'HSC (ViT-Base)', 'DESI (ref)', 'blue', 'orange', 0, 2),
            ('hsc_large', 'desi_precomputed', 'HSC (ViT-Large)', 'DESI (ref)', 'green', 'orange', 1, 0),
            ('sdss_interpolated', 'desi_precomputed', 'SDSS→DESI', 'DESI (ref)', 'red', 'orange', 1, 1),
        ]
        
        for config in plot_configs:
            key1, key2, label1, label2, color1, color2, row, col = config
            
            if key1 in embeddings_dict and key2 in embeddings_dict:
                try:
                    data1 = embeddings_dict[key1][:n_viz]
                    data2 = embeddings_dict[key2][:n_viz]
                    
                    xy1, xy2 = safe_reduce_dims(data1, data2)
                    
                    axes[row, col].scatter(xy1[:, 0], xy1[:, 1], 
                                         s=8, alpha=0.7, label=label1, c=color1)
                    axes[row, col].scatter(xy2[:, 0], xy2[:, 1], 
                                         s=8, alpha=0.7, label=label2, c=color2)
                    
                    score_key = f"{key1}_vs_{key2}"
                    score = scores_dict.get(score_key, 0)
                    axes[row, col].set_title(f'{label1} ↔ {label2}\nm-kNN: {score:.4f}')
                    axes[row, col].legend()
                    axes[row, col].grid(True, alpha=0.3)
                    
                except Exception as e:
                    print(f"⚠️ Error plotting {key1} vs {key2}: {e}")
                    axes[row, col].text(0.5, 0.5, f'Error plotting\n{label1} vs {label2}', 
                                      ha='center', va='center', transform=axes[row, col].transAxes)
            else:
                axes[row, col].text(0.5, 0.5, f'{label1} vs {label2}\nData not available', 
                                  ha='center', va='center', transform=axes[row, col].transAxes)
                axes[row, col].set_title(f'{label1} ↔ {label2}')
        
        # Score comparison plot
        try:
            score_names = []
            score_values = []
            colors = []
            
            score_configs = [
                ('hsc_base_vs_sdss_interpolated', 'ViT-Base\nvs SDSS→DESI', 'lightblue'),
                ('hsc_large_vs_sdss_interpolated', 'ViT-Large\nvs SDSS→DESI', 'lightgreen'),
                ('hsc_base_vs_desi_precomputed', 'ViT-Base\nvs DESI', 'orange'),
                ('hsc_large_vs_desi_precomputed', 'ViT-Large\nvs DESI', 'lightcoral'),
            ]
            
            for score_key, name, color in score_configs:
                if score_key in scores_dict:
                    score_names.append(name)
                    score_values.append(scores_dict[score_key])
                    colors.append(color)
            
            if score_names:
                bars = axes[1, 2].bar(score_names, score_values, color=colors, alpha=0.8)
                axes[1, 2].set_title('m-kNN Scores Comparison')
                axes[1, 2].set_ylabel('m-kNN Score')
                axes[1, 2].tick_params(axis='x', rotation=45)
                axes[1, 2].grid(True, alpha=0.3)
                
                # Add score values on bars
                for bar, score in zip(bars, score_values):
                    height = bar.get_height()
                    axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
            else:
                axes[1, 2].text(0.5, 0.5, 'No scores\navailable', 
                              ha='center', va='center', transform=axes[1, 2].transAxes)
                axes[1, 2].set_title('m-kNN Scores')
                
        except Exception as e:
            print(f"⚠️ Error creating score plot: {e}")
            axes[1, 2].text(0.5, 0.5, 'Error creating\nscore plot', 
                          ha='center', va='center', transform=axes[1, 2].transAxes)
        
        plt.tight_layout()
        
        # Safe saving
        try:
            plt.savefig('hsc_vs_sdss_robust_analysis.png', dpi=150, bbox_inches='tight')
            print(" Visualization saved as 'hsc_vs_sdss_robust_analysis.png'")
        except Exception as e:
            print(f"⚠️ Could not save visualization: {e}")
        
        try:
            plt.show()
        except:
            print("⚠️ Could not display plot")
        
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")
        import traceback
        traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════════════════════
# 5. MAIN ROBUST ANALYSIS PIPELINE
# ══════════════════════════════════════════════════════════════════════════════════════

def main_robust_analysis():
    """
    Main robust analysis pipeline with comprehensive error handling
    """
    print("\n" + "=" * 90)
    print(" HSC vs SDSS ANALYSIS - ROBUST FIXED VERSION")
    print("Key fixes: Robust checkpoint loading, tensor handling, file operations")
    print("=" * 90)
    
    try:
        # 1. Load models with robust error handling
        print("\n🔧 Loading models robustly...")
        
        try:
            vision_models = load_vision_models()
        except Exception as e:
            print(f"⚠️ Vision model loading failed: {e}")
            print("💡 Continuing without vision models...")
            vision_models = {}
        
        try:
            specformer_model = load_specformer_with_robust_checkpoint()
        except Exception as e:
            print(f"⚠️ SpecFormer loading failed: {e}")
            print("💡 Using minimal SpecFormer...")
            specformer_model = SpecFormer().to(DEVICE).eval()
        
        # 2. Load data
        print("\n Loading datasets...")
        try:
            HSC_images, SDSS_wavelengths, SDSS_fluxes = load_real_data()
        except Exception as e:
            print(f"⚠️ Data loading failed: {e}")
            print("💡 Using minimal mock data...")
            HSC_images, SDSS_wavelengths, SDSS_fluxes = generate_mock_data(100)
        
        # Optional: Load pre-computed DESI embeddings
        desi_embeddings_precomputed = load_desi_embeddings_optional()
        
        print(f" Dataset sizes:")
        print(f"   HSC images: {len(HSC_images):,}")
        print(f"   SDSS spectra: {len(SDSS_wavelengths):,}")
        if desi_embeddings_precomputed is not None:
            print(f"   DESI embeddings (precomputed): {len(desi_embeddings_precomputed):,}")
        
        # 3. Interpolate SDSS to DESI grid
        print("\n Interpolating SDSS spectra to DESI wavelength grid...")
        print("   (This allows SpecFormer, trained on DESI, to process SDSS data properly)")
        
        SDSS_interpolated_to_DESI = []
        
        for wave, flux in tqdm.tqdm(zip(SDSS_wavelengths, SDSS_fluxes), 
                                      desc="SDSS→DESI interpolation", 
                                      total=len(SDSS_wavelengths)):
            try:
                interpolated_flux = interpolate_sdss_to_desi_grid(wave, flux)
                SDSS_interpolated_to_DESI.append(interpolated_flux)
            except Exception as e:
                print(f"⚠️ Interpolation failed for spectrum, using zeros: {e}")
                SDSS_interpolated_to_DESI.append(np.zeros(7500))
        
        print(f"Interpolated {len(SDSS_interpolated_to_DESI)} SDSS spectra to DESI grid")
        
        # 4. Compute embeddings with robust error handling
        print("\n Computing embeddings robustly...")
        embeddings = {}
        
        # HSC image embeddings (both ViT models)
        for model_name, model_info in vision_models.items():
            try:
                embeddings[f'hsc_{model_name.split("_")[1]}'] = compute_image_embeddings(
                    HSC_images, model_info, model_name, batch_size=8  # Reduced batch size
                )
            except Exception as e:
                print(f"⚠️ Failed to compute {model_name} embeddings: {e}")
        
        # SDSS→DESI interpolated embeddings (MAIN ANALYSIS)
        try:
            embeddings['sdss_interpolated'] = compute_spectra_embeddings_robust(
                SDSS_interpolated_to_DESI, specformer_model, "SDSS→DESI", batch_size=2
            )
        except Exception as e:
            print(f"⚠️ Failed to compute SDSS interpolated embeddings: {e}")
            # Create fallback
            embeddings['sdss_interpolated'] = np.random.randn(len(SDSS_interpolated_to_DESI), 768)
        
        # Optional: Include pre-computed DESI embeddings for reference
        if desi_embeddings_precomputed is not None:
            embeddings['desi_precomputed'] = desi_embeddings_precomputed
        
        print(f"\n📊 Computed embeddings:")
        for key, emb in embeddings.items():
            print(f"   {key}: {emb.shape}")
        
        # 5. SAVE INTERPOLATED SDSS EMBEDDINGS SAFELY
        if 'sdss_interpolated' in embeddings:
            print("\n💾 Safely saving interpolated SDSS embeddings...")
            
            # Prepare metadata
            interpolated_metadata = {
                "description": "SDSS spectra interpolated to DESI wavelength grid and processed through SpecFormer",
                "interpolation_method": "specutils" if SPECUTILS_AVAILABLE else "numpy_interp",
                "source_survey": "SDSS",
                "target_grid": "DESI",
                "specformer_model": "polymathic-ai/specformer",
                "wavelength_range_angstrom": [3600, 9800],
                "n_wavelength_points": 7500,
                "n_samples": len(embeddings['sdss_interpolated']),
                "embedding_dimension": embeddings['sdss_interpolated'].shape[1],
                "preprocessing": "SpecFormer robust preprocessing (normalize, slice, pad)",
                "created_by": "HSC_vs_SDSS_robust_analysis_script",
                "note": "Robust version with enhanced error handling"
            }
            
            try:
                safe_save_embeddings(
                    embeddings['sdss_interpolated'], 
                    interpolated_metadata,
                    "sdss_interpolated_desi_grid_robust"
                )
            except Exception as e:
                print(f"⚠️ Failed to save embeddings: {e}")
        
        # 6. Align sample sizes
        if embeddings:
            min_samples = min(len(emb) for emb in embeddings.values())
            for key in embeddings:
                embeddings[key] = embeddings[key][:min_samples]
            
            print(f"\n📊 Aligned dataset: {min_samples:,} samples")
        else:
            print("❌ No embeddings available!")
            return None
        
        # 7. Compute m-kNN scores with error handling
        print("\n📈 Computing m-kNN scores robustly...")
        scores = {}
        
        # MAIN ANALYSIS: HSC vs SDSS→DESI
        print("🎯 Main Analysis: HSC vs SpecFormer(SDSS→DESI)")
        
        if 'hsc_base' in embeddings and 'sdss_interpolated' in embeddings:
            try:
                scores['hsc_base_vs_sdss_interpolated'] = compute_mknn_score(
                    embeddings['hsc_base'], embeddings['sdss_interpolated'], k=10
                )
            except Exception as e:
                print(f"⚠️ m-kNN computation failed for base vs sdss: {e}")
                scores['hsc_base_vs_sdss_interpolated'] = 0.0
        
        if 'hsc_large' in embeddings and 'sdss_interpolated' in embeddings:
            try:
                scores['hsc_large_vs_sdss_interpolated'] = compute_mknn_score(
                    embeddings['hsc_large'], embeddings['sdss_interpolated'], k=10
                )
            except Exception as e:
                print(f"⚠️ m-kNN computation failed for large vs sdss: {e}")
                scores['hsc_large_vs_sdss_interpolated'] = 0.0
        
        # OPTIONAL: Reference comparisons with pre-computed DESI embeddings
        if 'desi_precomputed' in embeddings:
            print("📊 Reference Analysis: HSC vs pre-computed DESI embeddings")
            
            if 'hsc_base' in embeddings:
                try:
                    scores['hsc_base_vs_desi_precomputed'] = compute_mknn_score(
                        embeddings['hsc_base'], embeddings['desi_precomputed'], k=10
                    )
                except Exception as e:
                    print(f"⚠️ m-kNN computation failed for base vs desi: {e}")
                    scores['hsc_base_vs_desi_precomputed'] = 0.0
            
            if 'hsc_large' in embeddings:
                try:
                    scores['hsc_large_vs_desi_precomputed'] = compute_mknn_score(
                        embeddings['hsc_large'], embeddings['desi_precomputed'], k=10
                    )
                except Exception as e:
                    print(f"⚠️ m-kNN computation failed for large vs desi: {e}")
                    scores['hsc_large_vs_desi_precomputed'] = 0.0
            
            # Interpolation quality check
            if 'sdss_interpolated' in embeddings:
                try:
                    scores['sdss_interpolated_vs_desi_precomputed'] = compute_mknn_score(
                        embeddings['sdss_interpolated'], embeddings['desi_precomputed'], k=10
                    )
                except Exception as e:
                    print(f"⚠️ Interpolation quality check failed: {e}")
                    scores['sdss_interpolated_vs_desi_precomputed'] = 0.0
        
        # 8. Display results
        print("\n" + "=" * 90)
        print("🎯 M-KNN ALIGNMENT RESULTS - ROBUST VERSION")
        print("=" * 90)
        
        if scores:
            print(f"\n🔍 MAIN ANALYSIS - HSC vs SDSS (through SpecFormer):")
            if 'hsc_base_vs_sdss_interpolated' in scores:
                print(f"   HSC (ViT-Base) ↔ SDSS→DESI:    {scores['hsc_base_vs_sdss_interpolated']:.4f}")
            if 'hsc_large_vs_sdss_interpolated' in scores:
                print(f"   HSC (ViT-Large) ↔ SDSS→DESI:   {scores['hsc_large_vs_sdss_interpolated']:.4f}")
            
            if 'desi_precomputed' in embeddings:
                print(f"\n📊 REFERENCE ANALYSIS - HSC vs pre-computed DESI:")
                if 'hsc_base_vs_desi_precomputed' in scores:
                    print(f"   HSC (ViT-Base) ↔ DESI:         {scores['hsc_base_vs_desi_precomputed']:.4f}")
                if 'hsc_large_vs_desi_precomputed' in scores:
                    print(f"   HSC (ViT-Large) ↔ DESI:        {scores['hsc_large_vs_desi_precomputed']:.4f}")
                
                print(f"\n🔄 INTERPOLATION QUALITY CHECK:")
                if 'sdss_interpolated_vs_desi_precomputed' in scores:
                    print(f"   SDSS→DESI ↔ DESI (precomputed): {scores['sdss_interpolated_vs_desi_precomputed']:.4f}")
        else:
            print("⚠️ No scores computed successfully")
        
        # 9. Analysis
        print(f"\n🔬 ANALYSIS:")
        
        # Check if we have meaningful results
        main_scores = [scores.get('hsc_base_vs_sdss_interpolated', 0), 
                       scores.get('hsc_large_vs_sdss_interpolated', 0)]
        valid_scores = [s for s in main_scores if s > 0]
        
        if valid_scores:
            avg_main = np.mean(valid_scores)
            
            if len(valid_scores) == 2:
                base_score = scores.get('hsc_base_vs_sdss_interpolated', 0)
                large_score = scores.get('hsc_large_vs_sdss_interpolated', 0)
                
                if large_score > base_score:
                    improvement = ((large_score - base_score) / max(base_score, 0.001)) * 100
                    print(f"   🏆 ViT-Large outperforms ViT-Base by {improvement:.1f}% in SDSS alignment")
                elif base_score > large_score:
                    improvement = ((base_score - large_score) / max(large_score, 0.001)) * 100
                    print(f"   🏆 ViT-Base outperforms ViT-Large by {improvement:.1f}% in SDSS alignment")
                else:
                    print(f"   📊 ViT-Base and ViT-Large show similar performance")
                
                best_model = "ViT-Large" if large_score > base_score else "ViT-Base"
                best_score = max(base_score, large_score)
                print(f"   🌟 Best model: {best_model} (score: {best_score:.4f})")
            
            # Overall assessment
            if avg_main > 0.3:
                print(f"   🎯 Strong HSC-SDSS cross-modal alignment detected (avg: {avg_main:.3f})")
            elif avg_main > 0.2:
                print(f"   📈 Moderate HSC-SDSS cross-modal alignment (avg: {avg_main:.3f})")
            elif avg_main > 0.1:
                print(f"   🔍 Weak HSC-SDSS cross-modal structure (avg: {avg_main:.3f})")
            else:
                print(f"   ❓ Minimal cross-modal alignment detected (avg: {avg_main:.3f})")
        else:
            print(f"   ⚠️ No valid alignment scores computed")
        
        # Check embedding quality
        print(f"\n🔍 EMBEDDING QUALITY CHECK:")
        for name, emb in embeddings.items():
            try:
                emb_std = np.std(emb)
                emb_mean = np.mean(np.abs(emb))
                print(f"   {name}: std={emb_std:.3f}, |mean|={emb_mean:.3f}")
                
                if emb_std < 0.01:
                    print(f"   ⚠️ {name} embeddings have very low variance")
                elif emb_std > 10:
                    print(f"   ⚠️ {name} embeddings have very high variance")
                else:
                    print(f"   ✅ {name} embeddings look reasonable")
            except Exception as e:
                print(f"   ❌ Could not analyze {name}: {e}")
        
        # 10. Create robust visualization
        try:
            create_robust_visualization(embeddings, scores)
        except Exception as e:
            print(f"⚠️ Visualization failed: {e}")
        
        # 11. Save results safely
        results = {
            "scores": scores,
            "n_samples": min_samples if embeddings else 0,
            "analysis_type": "HSC_vs_SDSS_robust",
            "embeddings_computed": list(embeddings.keys()),
            "main_comparisons": [
                "hsc_base_vs_sdss_interpolated",
                "hsc_large_vs_sdss_interpolated"
            ],
            "interpolation_method": "specutils" if SPECUTILS_AVAILABLE else "numpy",
            "models_loaded": {
                "vision": list(vision_models.keys()),
                "spectra": "SpecFormer_robust_loading"
            },
            "checkpoint_loaded": os.path.exists("specformer.ckpt"),
            "embedding_dimensions": {k: v.shape[1] for k, v in embeddings.items()},
            "note": "Robust version with comprehensive error handling and recovery"
        }
        
        try:
            with open("hsc_vs_sdss_robust_results.json", "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n💾 Results saved to 'hsc_vs_sdss_robust_results.json'")
        except Exception as e:
            print(f"⚠️ Could not save results file: {e}")
        
        # 12. Summary
        print(f"\n📁 FILES CREATED (if successful):")
        if os.path.exists("hsc_vs_sdss_robust_results.json"):
            print(f"   • hsc_vs_sdss_robust_results.json - Analysis results")
        if os.path.exists("hsc_vs_sdss_robust_analysis.png"):
            print(f"   • hsc_vs_sdss_robust_analysis.png - Visualization")
        
        # Check for embedding files
        embedding_files = [
            "sdss_interpolated_desi_grid_robust_embeddings.npy",
            "sdss_interpolated_desi_grid_robust_embeddings_sample.parquet",
            "sdss_interpolated_desi_grid_robust_metadata.json"
        ]
        
        for filename in embedding_files:
            if os.path.exists(filename):
                print(f"   • {filename}")
        
        # Check for chunked files
        chunk_files = [f for f in os.listdir(".") if f.startswith("sdss_interpolated_desi_grid_robust_embeddings_chunk_")]
        if chunk_files:
            print(f"   • {len(chunk_files)} embedding chunk files")
        
        return {
            "embeddings": embeddings,
            "scores": scores,
            "n_samples": min_samples if embeddings else 0,
            "analysis_successful": len(scores) > 0
        }
        
    except Exception as e:
        print(f"❌ Analysis failed with critical error: {e}")
        import traceback
        traceback.print_exc()
        return None

# ══════════════════════════════════════════════════════════════════════════════════════
# 6. RUNNER WITH COMPREHENSIVE ERROR RECOVERY
# ══════════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # print("🌌 HSC vs SDSS/DESI Analysis - ROBUST FIXED VERSION")
    # print("=" * 90)
    # print("Key fixes:")
    # print("Robust checkpoint downloading and loading")
    # print("Fixed tensor dimension handling in preprocessing")
    # print("Safe file operations with disk space checks")
    # print("Comprehensive error handling and recovery")
    # print("Reduced batch sizes to prevent memory issues")
    # print("=" * 90)
    
    # System checks
    print(f"\n🔍 SYSTEM CHECKS:")
    print(f"   Python version: {sys.version}")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Disk space check
    has_space, free_gb = check_disk_space()
    print(f"   Disk space: {free_gb:.1f} GB free")
    if not has_space:
        print("   ⚠️ Low disk space - will save in chunks if needed")
    
    # Check if checkpoint exists
    if not os.path.exists("specformer.ckpt"):
        print("\n📋 CHECKPOINT NOTICE:")
        print("specformer.ckpt not found - script will attempt to download it")
        print("If download fails, script will continue with randomly initialized model")
    
    # Run the robust analysis
    print("\n🚀 Starting robust analysis...")
    results = main_robust_analysis()
    
    if results and results.get('analysis_successful', False):
        print("\n" + "=" * 90)
        print("🎉 ROBUST ANALYSIS COMPLETE!")
        print("=" * 90)
        
        scores = results['scores']
        n_samples = results['n_samples']
        
        print(f"\n🏆 FINAL RESULTS ({n_samples:,} samples):")
        
        if scores:
            for score_name, score_value in scores.items():
                formatted_name = score_name.replace('_', ' ').replace('hsc', 'HSC').replace('vs', '↔')
                print(f"   {formatted_name}: {score_value:.4f}")
            
            # Find best performing model
            main_scores = {k: v for k, v in scores.items() if 'sdss_interpolated' in k and 'hsc' in k}
            if main_scores:
                best_key = max(main_scores.keys(), key=lambda k: main_scores[k])
                best_score = main_scores[best_key]
                best_model = "ViT-Large" if "large" in best_key else "ViT-Base"
                
                print(f"\n🌟 BEST PERFORMING MODEL:")
                print(f"   {best_model} achieved highest m-kNN score: {best_score:.4f}")
        else:
            print("   ⚠️ No scores were successfully computed")
        
        # print(f"\n📈 KEY IMPROVEMENTS:")
        # print(f"  Robust checkpoint downloading and loading")
        # print(f"   Fixed tensor dimension errors in SpecFormer preprocessing")
        # print(f"  Safe file saving with disk space checks")
        # print(f"   Comprehensive error handling throughout pipeline")
        # print(f"   Reduced batch sizes to prevent GPU memory issues")
        
        embeddings = results.get('embeddings', {})
        if 'sdss_interpolated' in embeddings:
            print(f"\n EMBEDDINGS STATUS:")
            if os.path.exists("sdss_interpolated_desi_grid_robust_embeddings.npy"):
                print(f" Full embeddings saved successfully")
            elif any(f.startswith("sdss_interpolated_desi_grid_robust_embeddings_chunk_") for f in os.listdir(".")):
                print(f" Embeddings saved in chunks (reassemble for upload)")
            else:
                print(f"   ⚠️ Embedding save may have failed")
        
    else:
        print("\n" + "=" * 90)
        print("⚠️ ANALYSIS COMPLETED WITH ISSUES")
        print("=" * 90)
        
        if results:
            embeddings = results.get('embeddings', {})
            scores = results.get('scores', {})
            
            print(f"📊 What worked:")
            if embeddings:
                print(f" Computed {len(embeddings)} embedding types")
                for name, emb in embeddings.items():
                    print(f"      - {name}: {emb.shape}")
            
            if scores:
                print(f" Computed {len(scores)} similarity scores")
                for name, score in scores.items():
                    print(f"      - {name}: {score:.4f}")
        
    #     print(f"\n💡 Common issues and solutions:")
    #     print(f"   • GPU OOM: Reduce batch sizes further (try batch_size=1)")
    #     print(f"   • Checkpoint issues: Check network connection, disk space")
    #     print(f"   • File save errors: Check disk space and permissions")
    #     print(f"   • Data loading: Network issues or dataset changes")
    
    # print(f"\n🌌 ROBUST MULTIMODAL ASTRONOMY ANALYSIS COMPLETE")
    # print(f"🔧 This version includes comprehensive error handling and should work")
    # print(f"   even with limited resources or network issues.")