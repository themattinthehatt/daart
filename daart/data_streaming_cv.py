#!/usr/bin/env python3
# daart/data_streaming.py

import os
import cv2
import numpy as np
import torch
from collections import OrderedDict

from daart.data import SingleDataset, load_label_csv, compute_sequences
from daart.transformer_loader_chunks import extract_patch_tokens_chunk


class StreamingSingleDataset(SingleDataset):
    """
    A SingleDataset that streams video -> ViT patch embeddings on‐the‐fly
    (never precomputing a giant features array), and presents them as
    the 'markers' signal to DAART.
    """

    def load_data(self, sequence_length: int, input_type: str):
        # 1) basic bookkeeping
        self.sequence_length = sequence_length
        self.input_type      = input_type  # typically 'features', but we map to 'markers'

        # 2) alias DAART‐passed transformer args
        self.config_path       = self.transformer_config
        self.checkpoint_path   = self.transformer_ckpt
        self.max_imgs_per_pass = getattr(self, 'max_imgs_per_pass', 256)

        # 3) load & window your one‐hot labels
        lbl_path = self.paths.get('labels_strong')
        if not lbl_path or not os.path.exists(lbl_path):
            raise FileNotFoundError(f"labels_strong file not found: {lbl_path!r}")
        labels_1hot, label_names = load_label_csv(lbl_path)
        labels_idx = labels_1hot.argmax(axis=1)

        lbl_seqs = compute_sequences(
            labels_idx,
            sequence_length,
            self.sequence_pad
        )

        # 4) stash labels in self.data
        self.data = OrderedDict()
        self.data['labels_strong'] = lbl_seqs
        self.label_names = label_names

        # 5) create dummy markers list so SingleDataset can infer n_sequences
        nseq = len(lbl_seqs)
        self.data['markers'] = [None] * nseq

        # 6) record your video path under 'markers'
        vid_path = self.paths.get('markers')
        if not vid_path or not os.path.exists(vid_path):
            raise FileNotFoundError(f"markers/video file not found: {vid_path!r}")
        self.video_path = vid_path

        # 7) defer setting input_size & feature_names until we know P*D
        self.input_size    = None
        self.feature_names = []

    def __len__(self):
        # now driven by the dummy 'markers' list
        return len(self.data['markers'])

    def __getitem__(self, idx: int) -> dict:
        # 1) read exactly sequence_length frames
        start = idx * self.sequence_length
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(self.sequence_length):
            ok, img = cap.read()
            if not ok:
                break
            frames.append(img)
        cap.release()

        # pad with last frame if needed
        if len(frames) < self.sequence_length:
            pad = frames[-1] if frames else np.zeros((224,224,3), np.uint8)
            frames += [pad] * (self.sequence_length - len(frames))

        # 2) run ViT‐patch extraction
        patches = extract_patch_tokens_chunk(
            frames,
            config_path       = self.config_path,
            checkpoint_path   = self.checkpoint_path,
            device            = self.device,
            max_imgs_per_pass = self.max_imgs_per_pass
        )  # → shape (L, P*D), NumPy array

        # 3) on first call, record scalar input_size = P*D
        if self.input_size is None:
            _, PD = patches.shape
            self.input_size    = PD
            self.feature_names = [f"patch_{i}" for i in range(PD)]

        # 4) fetch the stored label window
        labels_seq = self.data['labels_strong'][idx]  # shape = (L,)

        # 5) return the dict DAART expects
        return {
            'markers':       torch.from_numpy(patches).float(),   # (L, P*D)
            'labels_strong': torch.from_numpy(labels_seq).long(), # (L,)
            'batch_idx':     idx
        }
