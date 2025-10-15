# Vision Mamba (Vim) Integration Guide

This document provides information about the Vision Mamba integration into the Platonic Universe repository.

## Overview

Vision Mamba (Vim) is a state space model-based architecture for vision tasks. It uses efficient Mamba blocks instead of traditional self-attention mechanisms, making it a unique alternative to transformer-based models while maintaining strong performance on vision tasks.

## Available Models

The following Vision Mamba models are supported:

| Size  | Model ID                      | Hidden Dim | Parameters |
|-------|-------------------------------|------------|------------|
| Tiny  | hustvl/vim-tiny-midclstok     | 192        | ~7M        |
| Small | hustvl/vim-small-midclstok    | 384        | ~26M       |

## Usage

### Using the CLI

Run Vision Mamba experiments with the command-line interface:

```bash
# Test with JWST data
platonic_universe run --model vim --mode jwst

# Test with Legacy Survey data
platonic_universe run --model vim --mode legacysurvey

# Custom batch size and workers
platonic_universe run --model vim --mode jwst --batch-size 32 --num-workers 4
```

### Using Python API

```python
import pu

# Run experiment with Vision Mamba on JWST data
pu.run_experiment("vim", "jwst", batch_size=64, num_workers=2, knn_k=10)

# Run experiment with Vision Mamba on Legacy Survey data
pu.run_experiment("vim", "legacysurvey", batch_size=64, num_workers=2, knn_k=10)
```

## Implementation Details

### Embedding Extraction

Vision Mamba uses the **CLS token** for embedding extraction, similar to DINO models:
- Output shape: `(batch_size, num_tokens, hidden_dim)`
- Pooling strategy: `outputs[:, 0]` (first token)
- Final embedding shape: `(batch_size, hidden_dim)`

### Preprocessing

Vision Mamba uses a ViT-based preprocessing pipeline:
- Image processor: Uses ViT processor as fallback (`google/vit-base-patch16-224`)
- Preprocessor: `PreprocessHF` (same as ViT, DINO, etc.)
- Input image size: 224x224 (standard)
- Note: Vision Mamba models don't include a standard processor config, so we use ViT's processor which has the same input requirements

### Architecture

The adapter implementation:
```python
# In src/pu/models/hf.py
class HFAdapter(ModelAdapter):
    def load(self):
        if self.alias == "vim":
            # Use ViT processor as fallback
            self.processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
            self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).to("cuda").eval()
    
    def embed_for_mode(self, batch, mode):
        inputs = batch[f"{mode}"].to("cuda")
        with torch.no_grad():
            outputs = self.model(inputs).last_hidden_state
            if self.alias == "vim":
                # Vision Mamba uses CLS token similar to DINO
                emb = outputs[:, 0].detach()
        return emb
```

## Expected Output

When running experiments, Vision Mamba will generate:
1. **Embeddings**: Saved as `.parquet` files with columns like `vim_tiny_hsc`, `vim_tiny_jwst`
2. **MKNN Scores**: Mutual k-nearest neighbors scores measuring representational alignment
3. **Text Results**: MKNN scores saved to `data/{mode}_vim_mknn.txt`

Example output:
```
mknn vim, tiny: 0.32451234
mknn vim, small: 0.35678901
```

## Comparison with Other Models

Vision Mamba can be compared with other vision models in the repository:

| Model Type | Mechanism      | Pooling Strategy        | Supported Sizes        |
|------------|----------------|-------------------------|------------------------|
| ViT        | Attention      | Mean (excluding CLS)    | Base, Large, Huge      |
| DINOv2     | Attention      | CLS token               | Small, Base, Large, Giant |
| ConvNeXt   | Convolution    | Spatial mean            | Nano, Tiny, Base, Large |
| IJEPA      | Attention      | Mean over tokens        | Huge, Giant            |
| **Vim**    | **State Space**| **CLS token**           | **Tiny, Small**        |

## Testing

The integration includes comprehensive tests:

```bash
# Run integration tests
python /tmp/test_vim_integration.py

# Run embedding extraction tests
python /tmp/test_vim_embedding_extraction.py
```

All tests verify:
- ✓ Adapter registration
- ✓ Model loading structure
- ✓ Embedding extraction workflow
- ✓ Correct pooling strategy
- ✓ Preprocessor compatibility

## Requirements

Vision Mamba requires:
- PyTorch >= 2.0
- transformers >= 4.30.0
- CUDA-capable GPU (for full performance)
- Internet connection (for initial model download)

Models will be automatically downloaded from HuggingFace Hub on first use and cached locally.

## Troubleshooting

### Image Processor Error

If you encounter an error like `Can't load image processor for 'hustvl/vim-tiny-midclstok'`:
- **This is expected and handled automatically**
- Vision Mamba models don't include a standard processor configuration
- The adapter automatically uses a ViT processor as fallback (same input requirements)
- No action needed - the model will work correctly

### Model Download Issues

If you encounter download issues:
```python
# Set HuggingFace cache directory
import os
os.environ['HF_HOME'] = '/path/to/cache'
```

### Memory Issues

For large batches or limited GPU memory:
```bash
# Use smaller batch size
platonic_universe run --model vim --mode jwst --batch-size 16
```

### Trust Remote Code

Vision Mamba models require `trust_remote_code=True`. This is handled automatically by the adapter.

## References

- Vision Mamba Paper: [Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model](https://arxiv.org/abs/2401.09417)
- HuggingFace Model Hub: https://huggingface.co/hustvl
- Platonic Representation Hypothesis: See main README.md

## Contributing

To add more Vision Mamba variants:
1. Add model ID to `model_map` in `src/pu/experiments.py`
2. Add size string to the sizes list
3. Test with the provided test scripts

Example:
```python
"vim": (
    ["tiny", "small", "base"],  # Add "base"
    [
        "hustvl/vim-tiny-midclstok",
        "hustvl/vim-small-midclstok",
        "hustvl/vim-base-midclstok",  # Add new model
    ],
),
```
