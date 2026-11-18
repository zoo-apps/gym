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
            if not np.isfinite(part.scale) or part.scale < 0:
                raise ValueError(f"Invalid scale: {name}")
            bits = np.unpackbits(np.frombuffer(part.signs, dtype=np.uint8),
                                 count=reference.size, bitorder="little")
            signed = bits.astype(np.float32).reshape(reference.shape) * 2 - 1
            decoded.append(signed * np.float32(part.scale))
        ordered = np.sort(np.stack(decoded), axis=0)
        merged[name] = reference + ordered[trim:len(adapters) - trim].mean(axis=0)
    return merged
