#!/usr/bin/env python
"""
Simple script to load a SpecFormer model from checkpoint and extract embeddings from spectra.
"""

import torch
import numpy as np
from astroclip.models.specformer import SpecFormer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def _process_galaxy_wrapper(idx):
    spectra = np.array(idx["spectrum"]["flux"], dtype=np.float32)[..., np.newaxis]
    return {
        "spectra": spectra,
    }

def load_specformer_model(checkpoint_path):
    """Load SpecFormer model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model = SpecFormer(**checkpoint["hyper_parameters"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

# Example usage
if __name__ == "__main__":
    # Load model
    checkpoint_path = "specformer.ckpt"
    model = load_specformer_model(checkpoint_path).to("cuda:2")

    ds = (
      load_dataset("Smith42/desi_hsc_crossmatched", split="train", streaming=True)
      .select_columns(("spectrum"))
      .map(_process_galaxy_wrapper)
      .remove_columns(("spectrum"))
    )

    dl = iter(DataLoader(ds, batch_size=32, num_workers=0))

    # Example: Load your spectra data (replace with your actual data loading)
    # spectra_data should be shape (N, spectrum_length) or (N, spectrum_length, 1)
    # For DESI data, spectrum_length is typically 7781
    
    embeddings = []
    
    with torch.no_grad():
        for B in tqdm(dl, total=(18600//32)):
            S = B["spectra"].to("cuda:2")
            output = model(S)
            # Extract the embedding (not the reconstruction)
            batch_embeddings = output["embedding"].detach().cpu().numpy()
            embeddings.append(batch_embeddings[:, 1:, :].mean(axis=1))
    
    embeddings = np.concatenate(embeddings, axis=0)
    
    print(f"Output embeddings shape: {embeddings.shape}")
    
    # Save embeddings if needed
    np.save("spectra_embeddings.npy", embeddings)
