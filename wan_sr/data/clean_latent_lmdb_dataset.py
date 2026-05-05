from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset


class CleanLatentLMDBDataset(Dataset):
    """Dataset of clean LR/HR Wan latent pairs stored in sharded LMDB."""

    def __init__(
        self,
        data_dir: str | Path,
        dtype: torch.dtype = torch.float32,
        strict_channels: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.dtype = dtype
        self.strict_channels = strict_channels
        self.shards = self._discover_shards()
        if not self.shards:
            raise FileNotFoundError(f"No clean latent LMDB shards found under {self.data_dir}")

        self.envs: list[lmdb.Environment | None] = [None] * len(self.shards)
        self.index: list[tuple[int, int]] = []
        self.shard_meta: list[dict[str, Any]] = []
        for shard_id, shard_path in enumerate(self.shards):
            meta = self._read_shard_meta(shard_path)
            self.shard_meta.append(meta)
            for row_id in range(int(meta["num_samples"])):
                self.index.append((shard_id, row_id))

        if not self.index:
            raise FileNotFoundError(f"No clean latent samples found under {self.data_dir}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        shard_id, row_id = self.index[index]
        env = self._env(shard_id)
        meta = self.shard_meta[shard_id]

        z0_lr = _read_array(env, "z0_lr", row_id, np.float16, tuple(meta["z0_lr_shape"]))
        z0_hr = _read_array(env, "z0_hr", row_id, np.float16, tuple(meta["z0_hr_shape"]))
        z0_lr_tensor = torch.from_numpy(z0_lr.astype(np.float32, copy=False))
        z0_hr_tensor = torch.from_numpy(z0_hr.astype(np.float32, copy=False))

        if z0_lr_tensor.shape[0] != z0_hr_tensor.shape[0]:
            raise ValueError(f"channel mismatch at shard {self.shards[shard_id]} row {row_id}")
        if z0_lr_tensor.shape[1] != z0_hr_tensor.shape[1]:
            raise ValueError(f"latent time mismatch at shard {self.shards[shard_id]} row {row_id}")
        if self.strict_channels and z0_lr_tensor.shape[0] != 16:
            raise ValueError(f"expected Wan z_dim=16 at shard {self.shards[shard_id]} row {row_id}")
        if z0_hr_tensor.shape[-2] <= z0_lr_tensor.shape[-2] or z0_hr_tensor.shape[-1] <= z0_lr_tensor.shape[-1]:
            raise ValueError(f"expected HR spatial size > LR at shard {self.shards[shard_id]} row {row_id}")

        prompt = _read_text(env, "prompt", row_id, default="")
        row_meta = _read_text(env, "meta", row_id, default="{}")
        try:
            meta_json = json.dumps(json.loads(row_meta), ensure_ascii=False)
        except json.JSONDecodeError:
            meta_json = row_meta

        return {
            "z0_lr": z0_lr_tensor.to(self.dtype),
            "z0_hr": z0_hr_tensor.to(self.dtype),
            "sample_id": f"{self.shards[shard_id].name}:{row_id:06d}",
            "prompt": prompt,
            "meta_json": meta_json,
        }

    def _discover_shards(self) -> list[Path]:
        if not self.data_dir.exists():
            return []
        if (self.data_dir / "data.mdb").exists():
            return [self.data_dir]
        return [
            path
            for path in sorted(self.data_dir.iterdir())
            if path.is_dir() and (path / "data.mdb").exists()
        ]

    def _read_shard_meta(self, shard_path: Path) -> dict[str, Any]:
        env = lmdb.open(str(shard_path), readonly=True, lock=False, readahead=False, meminit=False, max_readers=256)
        try:
            with env.begin(write=False) as txn:
                raw = txn.get(b"metadata")
                if raw is not None:
                    meta = json.loads(raw.decode("utf-8"))
                else:
                    meta = {
                        "num_samples": int(_require(txn, "num_samples").decode("utf-8")),
                        "z0_lr_shape": _parse_shape(_require(txn, "z0_lr_shape")),
                        "z0_hr_shape": _parse_shape(_require(txn, "z0_hr_shape")),
                    }
        finally:
            env.close()
        return meta

    def _env(self, shard_id: int) -> lmdb.Environment:
        env = self.envs[shard_id]
        if env is None:
            env = lmdb.open(
                str(self.shards[shard_id]),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=256,
            )
            self.envs[shard_id] = env
        return env

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["envs"] = [None] * len(self.shards)
        return state

    def __del__(self) -> None:
        for env in getattr(self, "envs", []):
            if env is not None:
                env.close()


def _read_array(env: lmdb.Environment, name: str, row_id: int, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    key = f"{name}_{row_id:08d}_data".encode("utf-8")
    with env.begin(write=False) as txn:
        raw = txn.get(key)
    if raw is None:
        raise KeyError(f"missing LMDB key: {key.decode('utf-8')}")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def _read_text(env: lmdb.Environment, name: str, row_id: int, default: str = "") -> str:
    key = f"{name}_{row_id:08d}_data".encode("utf-8")
    with env.begin(write=False) as txn:
        raw = txn.get(key)
    if raw is None:
        return default
    return raw.decode("utf-8")


def _require(txn: lmdb.Transaction, key: str) -> bytes:
    raw = txn.get(key.encode("utf-8"))
    if raw is None:
        raise KeyError(f"missing LMDB metadata key: {key}")
    return raw


def _parse_shape(raw: bytes) -> list[int]:
    return [int(part) for part in raw.decode("utf-8").split()]
