"""Preprocessing cache: decode + window + letterbox each manifest image once.

Layout (`<cache>/<tier>/`):
  images.npy   uint8 (N, S, S) — letterboxed [0,1] frame × 255, one row per index entry;
               written with `np.lib.format.open_memmap` so readers memory-map it
  index.csv    image_id, row, path, ok, error + the per-image `input_statistics`
  cache.json   key = sha256(manifest sha256 ‖ preprocess config ‖ format), n, size, ...

The key ties a cache to the exact tier manifest and `PreprocessConfig` it was built from;
`FrameCache.open` refuses anything else (D-024). Rows the cache does not hold (a `--limit`
smoke cache, or a decode failure) fall back to on-the-fly decoding in `ManifestDataset`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
import torch
from numpy.typing import NDArray

from secondreader.core.registry import sha256_json
from secondreader.modalities.cxr.config import PreprocessConfig

log = structlog.get_logger(__name__)

CACHE_FORMAT = 1
STAT_KEYS: tuple[str, ...] = (
    "mean",
    "std",
    "p05",
    "p95",
    "entropy",
    "saturated_fraction",
    "src_h",
    "src_w",
    "aspect",
)
INDEX_COLUMNS: tuple[str, ...] = ("image_id", "row", "path", "ok", "error", *STAT_KEYS)


class StaleCacheError(RuntimeError):
    """The cache was built from a different manifest or preprocessing configuration."""


def cache_key(manifest_sha256: str, pre: PreprocessConfig) -> str:
    return sha256_json(
        {"manifest_sha256": manifest_sha256, "preprocess": pre.model_dump(), "format": CACHE_FORMAT}
    )


def _resolve(path: str, root: Path | None) -> Path:
    p = Path(path)
    if root is not None and not p.is_absolute():
        p = root / p
    return p


def _encode(path: Path, pre: PreprocessConfig) -> tuple[NDArray[np.uint8], dict[str, float]]:
    from secondreader.modalities.cxr.data import load_image
    from secondreader.modalities.cxr.preprocess import preprocess_frame

    pp = preprocess_frame(*load_image(path), pre)
    u8 = np.rint(np.clip(pp.image01, 0.0, 1.0) * 255.0).astype(np.uint8)
    return u8, pp.stats


def _init_worker() -> None:
    torch.set_num_threads(1)


def _encode_chunk(
    args: tuple[str, dict[str, Any], list[tuple[int, str]]],
) -> list[tuple[int, dict[str, float] | None, str]]:
    """Worker: decode a chunk of (row, path) and write rows straight into the memmap."""
    npy_path, pre_dict, items = args
    pre = PreprocessConfig.model_validate(pre_dict)
    mm = np.load(npy_path, mmap_mode="r+")
    out: list[tuple[int, dict[str, float] | None, str]] = []
    for row, path in items:
        try:
            u8, stats = _encode(Path(path), pre)
            mm[row] = u8
            out.append((row, stats, ""))
        except Exception as e:  # noqa: BLE001 — recorded per row, surfaced by the caller
            out.append((row, None, f"{type(e).__name__}: {e}"))
    mm.flush()
    del mm
    return out


def build_cache(
    df: pd.DataFrame,
    *,
    manifest_sha256: str,
    pre: PreprocessConfig,
    out_dir: Path,
    workers: int = 1,
    limit: int | None = None,
    root: Path | None = None,
    chunk_size: int = 32,
) -> dict[str, Any]:
    """Encode every manifest row (or the first `limit`) into `<out_dir>/images.npy`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = df.reset_index(drop=True)
    if limit is not None:
        rows = rows.iloc[:limit]
    n, size = len(rows), pre.image_size
    npy = out_dir / "images.npy"
    tmp_npy = out_dir / "images.npy.tmp"
    mm = np.lib.format.open_memmap(tmp_npy, mode="w+", dtype=np.uint8, shape=(n, size, size))
    mm.flush()
    del mm

    items = [(i, str(_resolve(p, root))) for i, p in enumerate(rows["path"].astype(str))]
    chunks = [
        (str(tmp_npy), pre.model_dump(), items[i : i + chunk_size]) for i in range(0, n, chunk_size)
    ]
    results: list[tuple[int, dict[str, float] | None, str]] = []
    done = 0
    if workers <= 1 or n <= chunk_size:
        _init_worker()
        for c in chunks:
            results.extend(_encode_chunk(c))
            done += len(c[2])
            if done % 1000 < chunk_size:
                log.info("cache_progress", done=done, n=n)
    else:
        ctx = get_context("spawn")
        with ctx.Pool(workers, initializer=_init_worker) as pool:
            for res in pool.imap_unordered(_encode_chunk, chunks):
                results.extend(res)
                done += len(res)
                if done % 1000 < chunk_size:
                    log.info("cache_progress", done=done, n=n)
    results.sort(key=lambda r: r[0])

    index = pd.DataFrame(
        {
            "image_id": rows["image_id"].astype(str).to_numpy(),
            "row": np.arange(n),
            "path": rows["path"].astype(str).to_numpy(),
            "ok": [int(s is not None) for _, s, _ in results],
            "error": [e for _, _, e in results],
            **{
                k: [float(s[k]) if s is not None else np.nan for _, s, _ in results]
                for k in STAT_KEYS
            },
        },
        columns=list(INDEX_COLUMNS),
    )
    n_failed = int((index["ok"] == 0).sum())
    if n_failed:
        log.warning(
            "cache_decode_failures", n_failed=n_failed, first=index.loc[index.ok == 0].head(3)
        )
    meta = {
        "format": CACHE_FORMAT,
        "key": cache_key(manifest_sha256, pre),
        "manifest_sha256": manifest_sha256,
        "preprocess": pre.model_dump(),
        "n": n,
        "n_manifest": int(len(df)),
        "limit": limit,
        "size": size,
        "dtype": "uint8",
        "n_failed": n_failed,
        "created_at": datetime.now(UTC).isoformat(),
    }
    index.to_csv(out_dir / "index.csv", index=False)
    tmp_npy.replace(npy)
    (out_dir / "cache.json").write_text(json.dumps(meta, indent=2))
    return meta


@dataclass
class FrameCache:
    """Read side of a tier cache. Safe to pickle into DataLoader workers (memmap opens lazily)."""

    dir: Path
    meta: dict[str, Any]
    index: pd.DataFrame
    _rows: dict[str, int] = field(default_factory=dict, repr=False)
    _mm: NDArray[np.uint8] | None = field(default=None, repr=False)

    @classmethod
    def open(
        cls, dir: Path, *, pre: PreprocessConfig, manifest_sha256: str | None = None
    ) -> FrameCache:
        dir = Path(dir)
        meta_path = dir / "cache.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"no cache at {dir} (missing cache.json)")
        meta: dict[str, Any] = json.loads(meta_path.read_text())
        if meta.get("format") != CACHE_FORMAT:
            raise StaleCacheError(f"{dir}: cache format {meta.get('format')} != {CACHE_FORMAT}")
        if meta.get("preprocess") != pre.model_dump():
            raise StaleCacheError(
                f"{dir}: built with preprocess {meta.get('preprocess')}, wanted {pre.model_dump()}"
            )
        if manifest_sha256 is not None and meta.get("key") != cache_key(manifest_sha256, pre):
            raise StaleCacheError(
                f"{dir}: built from manifest {str(meta.get('manifest_sha256'))[:12]}…, "
                f"current manifest is {manifest_sha256[:12]}… — rebuild with "
                "`secondreader cxr cache`"
            )
        index = pd.read_csv(dir / "index.csv", dtype={"image_id": str, "error": str})
        ok = index[index["ok"] == 1]
        rows = dict(zip(ok["image_id"].astype(str), ok["row"].astype(int), strict=True))
        return cls(dir=dir, meta=meta, index=index, _rows=rows)

    @property
    def size(self) -> int:
        return int(self.meta["size"])

    def __len__(self) -> int:
        return len(self._rows)

    def row_for(self, image_id: str) -> int | None:
        return self._rows.get(str(image_id))

    def _images(self) -> NDArray[np.uint8]:
        if self._mm is None:
            self._mm = np.load(self.dir / "images.npy", mmap_mode="r")
        return self._mm

    def image01(self, row: int) -> NDArray[np.float32]:
        """Letterboxed frame in [0,1] (quantized to 8 bits)."""
        return (np.asarray(self._images()[row], dtype=np.float32) / 255.0).astype(np.float32)

    def tensor(self, row: int) -> torch.Tensor:
        """(3, S, S) ImageNet-normalized tensor, matching `preprocess_frame(...).tensor`."""
        from secondreader.modalities.cxr.preprocess import IMAGENET_MEAN, IMAGENET_STD

        u8 = torch.from_numpy(np.array(self._images()[row]))  # copy out of the memmap
        g = u8.to(torch.float32).div_(255.0)[None].expand(3, -1, -1)
        mean = torch.from_numpy(IMAGENET_MEAN)[:, None, None]
        std = torch.from_numpy(IMAGENET_STD)[:, None, None]
        return (g - mean) / std

    def stats(self, row: int) -> dict[str, float]:
        r = self.index.iloc[row]
        return {k: float(r[k]) for k in STAT_KEYS}

    def __getstate__(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["_mm"] = None  # re-open the memmap inside each DataLoader worker
        return d


__all__ = [
    "CACHE_FORMAT",
    "FrameCache",
    "StaleCacheError",
    "build_cache",
    "cache_key",
]
