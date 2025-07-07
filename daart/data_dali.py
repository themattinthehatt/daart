import torch
from typing import Iterator, Tuple
from typing import Sequence, Union, Iterator, Tuple
import torch
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from daart.beast_utils.dataset_utils import build_file_lists, PerFrameLabelVideoPipeline, load_all_labels

def build_dali_data_loader(
    split: str,
    eids: Sequence[Union[int, str]],
    video_dir: str,
    label_dir: str,
    save_dir: str,
    hp: dict,
    device_id: int,
    world_size: int,
    pad: int
) -> DALIGenericIterator:
    """
    Build and return a DALI-based iterator for the given split.
    """
    assert split in ("train", "val", "test")

    # 1) Write the manifests
    build_file_lists(
        video_dir=video_dir,
        label_dir=label_dir,
        eids=eids,
        suffix=split,
        save_dir=save_dir
    )

    # 2) Build & compile the DALI pipeline
    pipe = PerFrameLabelVideoPipeline(
        video_file_list=f"{save_dir}/video_list_{split}.txt",
        batch_size=hp["batch_size"],
        num_threads=1,#hp["num_threads"],
        pad_last_batch=True,
        pad_sequences=True,     # *** enable padding at start/end of every window
        device_id=device_id,
        sequence_length=hp["sequence_length"],
        pad=pad,
        num_shards=world_size,
        shard_id=device_id,
        prefetch_queue_depth=2,
        random_shuffle=(split == "train")
    )
    pipe.build()

    # 3) Wrap in DALIGenericIterator
    return DALIGenericIterator(
        pipe,
        output_map=["frames", "frame_indices", "start_frame"],
        reader_name="VideoReader",
        auto_reset=True,
        last_batch_policy=LastBatchPolicy.PARTIAL,
    )


class FrameLabelLoader:
    def __init__(
        self,
        dali_loader,            # the DALIGenericIterator
        all_labels: torch.Tensor,  # tensor shape (num_videos, num_frames)
        pad: int,
        vit_cfg_path: str,
        device: torch.device,
        batch_size: int,
        eids
    ):
        self.loader     = dali_loader
        self.all_labels = all_labels
        self.pad        = pad
        self.n_tot_batches  = {'train': len(dali_loader)}
        self.device      = device
        self.batch_size  = batch_size
        self.eids        = eids

        # one‑time setup of ViT & PMA
        self._setup_vit(vit_cfg_path)
        
    def _setup_vit(self, vit_cfg_path: str) -> None:
        """Load YAML, init ViT backbone and PMA once."""
        import yaml
        from daart.model.pma_tcn import PMA
        from daart.model.vit_cm import ImageEncoderViTContrast

        with open(vit_cfg_path, "r") as f:
            self.vit_cfg = yaml.safe_load(f)
        self.vit_cfg["mask_ratio"] = 0

        # Initialize backbone
        self.model = ImageEncoderViTContrast(self.vit_cfg).to(self.device)
        self.model.vit_mae.from_pretrained("facebook/vit-mae-base")
        self.model.eval()

        # Initialize PMA
        self.pma = PMA(self.vit_cfg["hidden_size"], num_heads=8, num_seeds=1).to(self.device)

    def __len__(self):
        return self.n_tot_batches['train']
    
    def reset_iterators(self, dtype=None) -> None:
        # rewind the DALI loader
        self.loader.reset()
        # create a fresh Python iterator over it
        self._dali_iter = iter(self.loader)
        
    def extract_features(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
          frames: (B, Lp, 3, 224, 224) — padded clip
        Returns:
          feats:  (B, Lp, hidden_size) — ViT+PMA pooled features
        """
        B, Lp, C, H, W = frames.shape
        # ensure contiguous layout then flatten
        x = frames.contiguous().view(B * Lp, C, H, W).to(self.device)
        # ViT forward
        out = self.model(x)  # → (B*Lp, hidden_size, H', W')
        # reshape for PMA: (B*Lp, num_patches, hidden_size)
        out = out.permute(0, 2, 3, 1).reshape(
            B * Lp, -1, self.vit_cfg["hidden_size"]
        )
        # PMA pooling → (B*Lp, hidden_size)
        out = self.pma(out).squeeze(1)
        # restore sequence dimension → (B, Lp, hidden_size)
        feats = out.view(B, Lp, self.vit_cfg["hidden_size"])
        return feats

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns one batch:
          feats  (B, Lp, hidden_size),
          labels (B, Lp, 1),
          fidx   (B,),
          start  (B,)
        """
        batch = next(self._dali_iter)[0]
        frames = batch["frames"]            # (B, Lp, 3,224,224)
        fidx   = batch["frame_indices"]     # (B,)
        start  = batch["start_frame"]       # (B,)
    
        B, Lp, _, _, _ = frames.shape
    
        # gather labels over full padded window
        seq_idxs = torch.arange(Lp, device=self.device)[None, :]          # (1, Lp)
        time_idx = start[:, None] + seq_idxs                              # (B, Lp)
        vid_idx  = fidx[:, None].expand_as(time_idx)                      # (B, Lp)
        raw_lbl  = self.all_labels[vid_idx, time_idx].long()              # (B, Lp)
        labels   = raw_lbl.view(B, Lp, 1)                                 # (B, Lp, 1)
    
        # extract features under no_grad to save memory
        with torch.no_grad():
            feats = self.extract_features(frames)                         # (B, Lp, hidden_size)
    
        return feats, labels, fidx, start

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convenience method to fetch one batch.
        """
        feats, labels, fidx, start =  self.__next__()
        return {'markers': feats, 'labels_strong': labels, 'fidx': fidx, 'start': start}, -1

    # def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
    #     # Pull one batch from the DALI iterator
    #     batch = next(self._dali_iter)[0]
    #     frames     = batch["frames"]         # (B, Lp, 3, 224, 224)
    #     fidx       = batch["frame_indices"]  # (B,)
    #     start      = batch["start_frame"]    # (B,)

    #     B, Lp, C, H, W = frames.shape

    #     # 1) Compute labels over the full padded window
    #     seq_idxs = torch.arange(Lp, device=self.device)          # (Lp,)
    #     time_idx = start[:, None] + seq_idxs[None, :]           # (B, Lp)
    #     vid_idx  = fidx[:, None].expand_as(time_idx)            # (B, Lp)
    #     raw_lbl  = self.all_labels[vid_idx, time_idx].long()   # (B, Lp)
    #     labels = raw_lbl.view(B, Lp, 1)                   # (B, Lp, 1)

    #     # 2) Extract ViT+PMA features over the same window
    #     feats = self.extract_features(frames)                    # (B, Lp, hidden_size)

    #     return feats, labels

    # def extract_features(self, frames: torch.Tensor) -> torch.Tensor:
    #     """
    #     Inputs:
    #       frames: (B, Lp, 3, 224, 224), where Lp = sequence_length + 2*pad
    #     Returns:
    #       feats:  (B, Lp, hidden_size)
    #     """
    #     B, Lp, C, H, W = frames.shape

    #     # a) ensure contiguous and flatten for ViT
    #     x = frames.contiguous().view(B * Lp, C, H, W).to(self.device)   # (B*Lp, 3,224,224)

    #     # b) forward through ViT backbone
    #     with torch.no_grad():
    #         out = self.model(x)                                            # (B*Lp, hidden_size, H',W')

    #     # c) reshape to (B*Lp, num_patches, hidden_size)
    #     out = out.permute(0, 2, 3, 1).reshape(
    #         B * Lp, -1, self.vit_cfg["hidden_size"]
    #     )

    #     # d) PMA pooling → (B*Lp, hidden_size)
    #     out = self.pma(out).squeeze(1)

    #     # e) restore sequence dimension → (B, Lp, hidden_size)
    #     feats = out.view(B, Lp, self.vit_cfg["hidden_size"])
    #     return feats
        
    # def next_batch(self, dtype=None, transforms=None):
    #     feats, labels = self.__next__()
    #     return {'markers': feats, 'labels_strong': labels}, -1