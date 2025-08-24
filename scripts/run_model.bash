#!/bin/bash

MODEL=$1
GPU=$2

CUDA_VISIBLE_DEVICES=$GPU uv run scripts/get_embs.py --model $MODEL --mode jwst --num-workers 32
CUDA_VISIBLE_DEVICES=$GPU uv run scripts/get_embs.py --model $MODEL --mode sdss
CUDA_VISIBLE_DEVICES=$GPU uv run scripts/get_embs.py --model $MODEL --mode desi
CUDA_VISIBLE_DEVICES=$GPU uv run scripts/get_embs.py --model $MODEL --mode legacysurvey --num-workers 32
