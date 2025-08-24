#!/bin/bash

GPU=$1

CUDA_VISIBLE_DEVICES=$GPU uv run scripts/get_astropt_embs.py --mode jwst --num-workers 32
CUDA_VISIBLE_DEVICES=$GPU uv run scripts/get_astropt_embs.py --mode sdss
CUDA_VISIBLE_DEVICES=$GPU uv run scripts/get_astropt_embs.py --mode desi
CUDA_VISIBLE_DEVICES=$GPU uv run scripts/get_astropt_embs.py --mode legacysurvey --num-workers 32
