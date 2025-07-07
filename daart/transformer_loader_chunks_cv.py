#!/usr/bin/env python3
# transformer_loader.py

import os
import cv2
import yaml
import torch
import numpy as np
import time
import warnings

from functools import lru_cache
from contextlib import contextmanager
from torch.cuda.amp import autocast
from daart.model.vit_mae.vit_mae import ImageEncoderViTMAE

warnings.filterwarnings("ignore")

# ─── TIMER CONTEXT MANAGER ────────────────────────────────────────────
@contextmanager
def timer(name: str):
    t0 = time.time()
    yield
    print(f"[TIMER] {name:30s}: {time.time() - t0:.3f}s")
# ────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=2)
def _load_model(config_path: str, checkpoint_path: str, device: str):
    """
    Load & prepare the ViT-MAE model. Cached so we only ever do this once.
    """
    with timer("load config & build model"):
        cfg = yaml.safe_load(open(config_path))
        cfg['mask_ratio'] = 0.0
        model = ImageEncoderViTMAE(config=cfg)
        model.vit_mae.from_pretrained("facebook/vit-mae-base")

    with timer("load checkpoint"):
        if checkpoint_path and os.path.isfile(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            filtered = {
                k.replace('vit_mae.', ''): v
                for k, v in ckpt.items()
                if k.startswith('vit_mae.')
                and k.replace('vit_mae.', '') in model.vit_mae.state_dict()
            }
            model.vit_mae.load_state_dict(filtered, strict=False)

    with timer("to device & eval"):
        model = model.to(device)
        model.eval()
        # use half precision on GPU
        if device.startswith('cuda'):
            model.half()

    return model


def extract_patch_tokens_chunk(
    frames: list,
    config_path: str,
    checkpoint_path: str,
    device: str = "cuda",
    max_imgs_per_pass: int = 256,
) -> np.ndarray:
    """
    Given a list of raw BGR frames, runs them through ViT-MAE and returns
    a NumPy array of shape (T, P*D) with all patch embeddings (no CLS token).
    """
    # 1) Load (or fetch cached) model
    model = _load_model(config_path, checkpoint_path, device)
    device = torch.device(device)

    # 2) Resize, convert to RGB, stack into a NumPy array
    arr = np.stack([
        cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB)
        for frame in frames
    ], axis=0)  # shape = (T, H, W, C)

    # 3) To tensor & normalize
    x = (
        torch.from_numpy(arr)
             .permute(0, 3, 1, 2)
             .float()
             .div(255.0)
             .to(device)
    )
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    x = (x - mean) / std  # shape = (T, C, H, W)

    # 4) Inference in sub‐batches
    outs = []
    for i in range(0, x.shape[0], max_imgs_per_pass):
        sub = x[i : i + max_imgs_per_pass]
        if device.type == "cuda":
            torch.cuda.synchronize()
        with torch.no_grad(), autocast():
            out = model(sub)           # shape = (sub_size, P+1, D)
        if device.type == "cuda":
            torch.cuda.synchronize()
        outs.append(out.cpu())

    out = torch.cat(outs, dim=0)       # shape = (T, P+1, D)

    # 5) Drop the CLS token and flatten to (T, P*D)
    patches = out[:, 1:, :].reshape(out.shape[0], -1).numpy()
    return patches
