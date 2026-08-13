from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def _key(name: str, row_id: int) -> bytes:
    return f"{name}_{row_id:08d}".encode("utf-8")


class UniversalCleanLatentDataset(Dataset):
    """Read multi-scale clean latent pairs from schema ``wan_uni_clean_v1``.

    A sample contains one HR latent and all available LR variants.  ``set_epoch``
    deterministically rotates the selected scale so distributed training does
    not depend on worker-local random state.  Use ``ScaleBucketBatchSampler``
    when batch_size is larger than one because different scales have different
    source tensor shapes.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        scales: Sequence[str | float] | None = None,
        strict_channels: bool = True,
        dtype: torch.dtype = torch.float32,
        seed: int = 1234,
    ) -> None:
        self.root = Path(root)
        self.shards = sorted(path for path in self.root.glob("shard_*") if path.is_dir())
        if not self.shards:
            raise FileNotFoundError(f"No shard_* directories found under {self.root}")
        self.strict_channels = bool(strict_channels)
        self.dtype = dtype
        self.seed = int(seed)
        self._envs: dict[int, lmdb.Environment] = {}
        self.shard_meta: list[dict[str, Any]] = []
        self.index: list[tuple[int, int]] = []

        requested = {_canonical_scale(scale) for scale in scales} if scales is not None else None
        common_scales: set[str] | None = None
        for shard_id, shard in enumerate(self.shards):
            meta = self._read_shard_meta(shard)
            if meta.get("schema") != "wan_uni_clean_v1":
                raise ValueError(f"Unsupported schema in {shard}: {meta.get('schema')!r}")
            available = {_canonical_scale(scale) for scale in meta["lr_shapes"]}
            if requested is not None and not requested.issubset(available):
                raise ValueError(f"Shard {shard} is missing requested scales {sorted(requested - available)}")
            common_scales = available if common_scales is None else common_scales & available
            self.shard_meta.append(meta)
            self.index.extend((shard_id, row_id) for row_id in range(int(meta["num_samples"])))

        selected = requested or common_scales or set()
        if not selected:
            raise ValueError("Dataset has no common LR scales")
        self.scales = tuple(sorted(selected, key=float))

    def __len__(self) -> int:
        return len(self.index) * len(self.scales)

    @property
    def num_source_samples(self) -> int:
        return len(self.index)

    def virtual_indices_for_sources(self, source_indices: Sequence[int]) -> list[int]:
        return [source_index * len(self.scales) + offset for source_index in source_indices for offset in range(len(self.scales))]

    def set_epoch(self, epoch: int) -> None:
        # The virtual index already expands every source sample across every
        # scale. Epoch affects sampler shuffling, not sample tensor shape.
        _ = epoch

    def selected_scale(self, index: int) -> str:
        return self.scales[index % len(self.scales)]

    def physical_index(self, index: int) -> int:
        return index // len(self.scales)

    def bucket_key(self, index: int) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
        shard_id, _ = self.index[self.physical_index(index)]
        meta = self.shard_meta[shard_id]
        scale = self.selected_scale(index)
        return scale, tuple(meta["lr_shapes"][scale]), tuple(meta["hr_shape"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        shard_id, row_id = self.index[self.physical_index(index)]
        meta = self.shard_meta[shard_id]
        env = self._get_env(shard_id)
        scale = self.selected_scale(index)
        hr_shape = tuple(meta["hr_shape"])
        stored_scale = next(name for name in meta["lr_shapes"] if _canonical_scale(name) == scale)
        lr_shape = tuple(meta["lr_shapes"][stored_scale])
        with env.begin() as txn:
            z0_hr = _read_array(txn, "z0_hr", row_id, hr_shape)
            z0_lr = _read_array(txn, f"z0_lr_{_scale_key(scale)}", row_id, lr_shape)
            prompt = _read_text(txn, "prompt", row_id, default="")
            sample_meta = _read_text(txn, "meta", row_id, default="{}")

        z0_lr_tensor = torch.from_numpy(z0_lr.astype(np.float32, copy=False)).to(self.dtype)
        z0_hr_tensor = torch.from_numpy(z0_hr.astype(np.float32, copy=False)).to(self.dtype)
        if self.strict_channels and (z0_lr_tensor.shape[0] != 16 or z0_hr_tensor.shape[0] != 16):
            raise ValueError(f"Wan latents must have 16 channels, got {lr_shape} and {hr_shape}")
        if z0_lr_tensor.shape[:2] != z0_hr_tensor.shape[:2]:
            raise ValueError("LR/HR latents must share channel and temporal dimensions")

        scale_h = z0_hr_tensor.shape[-2] / z0_lr_tensor.shape[-2]
        scale_w = z0_hr_tensor.shape[-1] / z0_lr_tensor.shape[-1]
        return {
            "z0_lr": z0_lr_tensor,
            "z0_hr": z0_hr_tensor,
            "scale": torch.tensor(float(scale), dtype=torch.float32),
            "scale_hw": torch.tensor([scale_h, scale_w], dtype=torch.float32),
            "source_size": torch.tensor(z0_lr_tensor.shape[-2:], dtype=torch.int64),
            "target_size": torch.tensor(z0_hr_tensor.shape[-2:], dtype=torch.int64),
            "scale_key": scale,
            "prompt": prompt,
            "meta_json": sample_meta,
        }

    def _get_env(self, shard_id: int) -> lmdb.Environment:
        env = self._envs.get(shard_id)
        if env is None:
            env = lmdb.open(
                str(self.shards[shard_id]),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=256,
            )
            self._envs[shard_id] = env
        return env

    @staticmethod
    def _read_shard_meta(path: Path) -> dict[str, Any]:
        env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, meminit=False)
        try:
            with env.begin() as txn:
                raw = txn.get(b"metadata")
                if raw is None:
                    raise KeyError(f"Missing metadata in {path}")
                return json.loads(raw.decode("utf-8"))
        finally:
            env.close()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_envs"] = {}
        return state

    def __del__(self) -> None:
        for env in getattr(self, "_envs", {}).values():
            env.close()


class ScaleBucketBatchSampler(Sampler[list[int]]):
    """Batch indices with identical source/target tensor shapes."""

    def __init__(
        self,
        dataset: UniversalCleanLatentDataset,
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 1234,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if batch_size < 1 or world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("invalid batch/distributed sampler parameters")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.dataset.set_epoch(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        buckets: dict[tuple[str, tuple[int, ...], tuple[int, ...]], list[int]] = defaultdict(list)
        for index in range(len(self.dataset)):
            buckets[self.dataset.bucket_key(index)].append(index)

        batches: list[list[int]] = []
        for indices in buckets.values():
            if self.shuffle:
                rng.shuffle(indices)
            global_batch = self.batch_size * self.world_size
            usable = len(indices) - (len(indices) % global_batch) if self.drop_last else len(indices)
            for start in range(0, usable, global_batch):
                group = indices[start : start + global_batch]
                local = group[self.rank * self.batch_size : (self.rank + 1) * self.batch_size]
                if len(local) == self.batch_size or (local and not self.drop_last):
                    batches.append(local)
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        counts: dict[tuple[str, tuple[int, ...], tuple[int, ...]], int] = defaultdict(int)
        for index in range(len(self.dataset)):
            counts[self.dataset.bucket_key(index)] += 1
        global_batch = self.batch_size * self.world_size
        if self.drop_last:
            return sum(count // global_batch for count in counts.values())
        return sum((count + global_batch - 1) // global_batch for count in counts.values())


def _canonical_scale(value: str | float) -> str:
    return f"{float(value):g}"


def _scale_key(scale: str) -> str:
    return scale.replace(".", "p")


def _read_array(txn: lmdb.Transaction, name: str, row_id: int, shape: tuple[int, ...]) -> np.ndarray:
    raw = txn.get(_key(name, row_id))
    if raw is None:
        raise KeyError(f"Missing LMDB key {_key(name, row_id)!r}")
    return np.frombuffer(raw, dtype=np.float16).reshape(shape).copy()


def _read_text(txn: lmdb.Transaction, name: str, row_id: int, *, default: str) -> str:
    raw = txn.get(_key(name, row_id))
    return default if raw is None else raw.decode("utf-8")
