"""Pack compatible float32 model updates and aggregate their decoded coordinates."""

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np


def tensor_map_digest(tensors, schema):
    """Bind semantic conventions, tensor names, shapes and little-endian values."""
    digest = hashlib.sha256()
    layout = []
    for name in sorted(tensors):
        value = tensors[name]
        if value.dtype != np.float32 or not value.size or not np.isfinite(value).all():
            raise ValueError(f"Expected a nonempty, finite float32 tensor: {name}")
        layout.append([name, value.shape])
    digest.update(json.dumps([schema, layout], separators=(",", ":")).encode() + b"\0")
    for name in sorted(tensors):
        digest.update(tensors[name].astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class PackedTensor:
    scale: float
    signs: bytes


@dataclass
class Adapter:
    parent: str
    tensors: dict[str, PackedTensor]

    def digest(self):
        digest = hashlib.sha256(bytes.fromhex(self.parent))
        for name in sorted(self.tensors):
            tensor = self.tensors[name]
            if not 0 <= tensor.scale <= float(np.finfo(np.float32).max):
                raise ValueError(f"Invalid scale: {name}")
            digest.update(json.dumps([name, len(tensor.signs)]).encode() + b"\0")
            digest.update(np.asarray(tensor.scale, dtype="<f4").tobytes())
            digest.update(tensor.signs)
        return digest.hexdigest()


def pack(base, tuned, schema):
    parent = tensor_map_digest(base, schema)
    if base.keys() != tuned.keys():
        raise ValueError("Tensor names differ")
    tensors = {}
    for name, reference in base.items():
        value = tuned[name]
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError(f"Tensor shape or dtype differs: {name}")
        delta = value - reference
        if not np.isfinite(delta).all():
            raise ValueError(f"Nonfinite delta: {name}")
        scale = float(np.float32(np.abs(delta).mean(dtype=np.float64)))
        signs = np.packbits(delta.ravel() >= 0, bitorder="little").tobytes()
        tensors[name] = PackedTensor(scale, signs)
    return Adapter(parent, tensors)


def merge(base, adapters, schema, trim_fraction):
    if not adapters or not 0 <= trim_fraction < 0.5:
        raise ValueError("Need adapters and a trim fraction in [0, 0.5)")
    parent = tensor_map_digest(base, schema)
    if any(a.parent != parent or a.tensors.keys() != base.keys() for a in adapters):
        raise ValueError("Incompatible parent, schema or tensor names")
    if len({a.digest() for a in adapters}) != len(adapters):
        raise ValueError("Duplicate adapter")
    trim = math.floor(len(adapters) * trim_fraction)
    merged = {}
    for name, reference in base.items():
        decoded = []
        for adapter in adapters:
            part = adapter.tensors[name]
            if len(part.signs) != (reference.size + 7) // 8:
                raise ValueError(f"Packed length mismatch: {name}")
            tail = reference.size % 8
            if tail and part.signs[-1] >> tail:
                raise ValueError(f"Nonzero padding bits: {name}")
            bits = np.unpackbits(np.frombuffer(part.signs, dtype=np.uint8),
                                 count=reference.size, bitorder="little")
            signed = bits.astype(np.float32).reshape(reference.shape) * 2 - 1
            decoded.append(signed * np.float32(part.scale))
        ordered = np.sort(np.stack(decoded), axis=0)
        kept = ordered[trim:len(adapters) - trim]
        with np.errstate(over="ignore", invalid="ignore"):
            result = reference + kept.mean(axis=0)
        overflow = ~np.isfinite(result)
        if np.any(overflow):
            # A float32 sum can overflow even when its mean fits. Recompute
            # only affected coordinates to preserve rounding everywhere else.
            recovered = reference[overflow].astype(np.float64) + kept[:, overflow].mean(
                axis=0, dtype=np.float64
            )
            if np.any(np.abs(recovered) > float(np.finfo(np.float32).max)):
                raise ValueError(f"Merged tensor exceeds float32 range: {name}")
            result = np.asarray(result)
            result[overflow] = recovered
        merged[name] = result
    return merged
