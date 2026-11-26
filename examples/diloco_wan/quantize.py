# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zoo Labs Foundation and Hanzo AI Contributors.
"""Wire-serialized pseudo-gradient compression codecs for WAN synchronization.

Every codec encodes a flat float pseudo-gradient vector into real binary wire
bytes with a deterministic header (magic, version, codec ID, block size, length)
so WAN byte accounting reflects exact transmitted payloads rather than estimates.
"""

from __future__ import annotations

import math
import struct
from typing import List, Sequence, Tuple

WIRE_MAGIC = b"DLCW"
WIRE_VERSION = 1

CODEC_FP32 = "fp32"
CODEC_INT8_BLOCK = "int8_block"
CODEC_INT4_BLOCK = "int4_block"
CODEC_SIGN_1BIT_NO_EF = "sign_1bit_no_ef"

_CODEC_TO_ID = {
    CODEC_FP32: 1,
    CODEC_INT8_BLOCK: 2,
    CODEC_INT4_BLOCK: 3,
    CODEC_SIGN_1BIT_NO_EF: 4,
}
_ID_TO_CODEC = {v: k for k, v in _CODEC_TO_ID.items()}

# Header: 4B magic, 1B version, 1B codec_id, 2B block_size, 4B num_elements (12B)
_HEADER_STRUCT = struct.Struct("<4sBBHI")


def encode_tensor(
    values: Sequence[float],
    codec: str = CODEC_FP32,
    block_size: int = 64,
) -> bytes:
    """Serialize a 1D float vector into real wire bytes using the specified codec."""
    if codec not in _CODEC_TO_ID:
        raise ValueError(f"Unknown codec: {codec}")
    n = len(values)
    codec_id = _CODEC_TO_ID[codec]
    header = _HEADER_STRUCT.pack(WIRE_MAGIC, WIRE_VERSION, codec_id, block_size, n)

    if codec == CODEC_FP32:
        body = struct.pack(f"<{n}f", *values)
        return header + body

    if codec == CODEC_INT8_BLOCK:
        num_blocks = (n + block_size - 1) // block_size
        scales = [0.0] * num_blocks
        qbytes = bytearray(n)
        for b in range(num_blocks):
            start = b * block_size
            end = min(n, start + block_size)
            max_abs = 0.0
            for i in range(start, end):
                v = abs(values[i])
                if not math.isfinite(v):
                    raise ValueError("Cannot quantize non-finite value in int8_block")
                if v > max_abs:
                    max_abs = v
            scale = max_abs / 127.0 if max_abs > 0.0 else 0.0
            scales[b] = scale
            if scale > 0.0:
                inv_scale = 1.0 / scale
                for i in range(start, end):
                    q = int(round(values[i] * inv_scale))
                    if q > 127:
                        q = 127
                    elif q < -127:
                        q = -127
                    qbytes[i] = q & 0xFF
        scale_bytes = struct.pack(f"<{num_blocks}f", *scales)
        return header + scale_bytes + bytes(qbytes)

    if codec == CODEC_INT4_BLOCK:
        num_blocks = (n + block_size - 1) // block_size
        scales = [0.0] * num_blocks
        packed = bytearray((n + 1) // 2)
        for b in range(num_blocks):
            start = b * block_size
            end = min(n, start + block_size)
            max_abs = 0.0
            for i in range(start, end):
                v = abs(values[i])
                if not math.isfinite(v):
                    raise ValueError("Cannot quantize non-finite value in int4_block")
                if v > max_abs:
                    max_abs = v
            scale = max_abs / 7.0 if max_abs > 0.0 else 0.0
            scales[b] = scale
            if scale > 0.0:
                inv_scale = 1.0 / scale
                for i in range(start, end):
                    q = int(round(values[i] * inv_scale))
                    if q > 7:
                        q = 7
                    elif q < -7:
                        q = -7
                    nibble = (q + 7) & 0x0F  # map [-7..7] -> [0..14]
                    byte_idx = i >> 1
                    if (i & 1) == 0:
                        packed[byte_idx] |= nibble
                    else:
                        packed[byte_idx] |= nibble << 4
            else:
                # Zero block maps to nibble 7 (representing signed 0)
                for i in range(start, end):
                    byte_idx = i >> 1
                    if (i & 1) == 0:
                        packed[byte_idx] |= 7
                    else:
                        packed[byte_idx] |= 7 << 4
        scale_bytes = struct.pack(f"<{num_blocks}f", *scales)
        return header + scale_bytes + bytes(packed)

    if codec == CODEC_SIGN_1BIT_NO_EF:
        # Ultra-aggressive 1-bit sign quantization without error feedback.
        # Global mean-absolute magnitude scale + 1 bit per coordinate.
        abs_sum = 0.0
        for v in values:
            if not math.isfinite(v):
                raise ValueError("Cannot quantize non-finite value in sign_1bit_no_ef")
            abs_sum += abs(v)
        mean_abs = abs_sum / n if n > 0 else 0.0
        packed = bytearray((n + 7) // 8)
        for i, v in enumerate(values):
            if v >= 0.0:
                packed[i >> 3] |= 1 << (i & 7)
        return header + struct.pack("<f", mean_abs) + bytes(packed)

    raise AssertionError("Unreachable")


def decode_tensor(payload: bytes) -> Tuple[str, List[float]]:
    """Deserialize binary wire bytes back into (codec_name, reconstructed_floats)."""
    if len(payload) < _HEADER_STRUCT.size:
        raise ValueError("Payload truncated before wire header")
    magic, version, codec_id, block_size, n = _HEADER_STRUCT.unpack_from(payload, 0)
    if magic != WIRE_MAGIC or version != WIRE_VERSION:
        raise ValueError("Invalid DiLoCo wire magic or version")
    if codec_id not in _ID_TO_CODEC:
        raise ValueError(f"Unknown wire codec_id: {codec_id}")
    codec = _ID_TO_CODEC[codec_id]
    offset = _HEADER_STRUCT.size

    if codec == CODEC_FP32:
        expected = offset + 4 * n
        if len(payload) != expected:
            raise ValueError(f"FP32 payload length mismatch: {len(payload)} != {expected}")
        return codec, list(struct.unpack_from(f"<{n}f", payload, offset))

    if codec == CODEC_INT8_BLOCK:
        num_blocks = (n + block_size - 1) // block_size
        scales = struct.unpack_from(f"<{num_blocks}f", payload, offset)
        offset += 4 * num_blocks
        qbytes = payload[offset : offset + n]
        if len(qbytes) != n:
            raise ValueError("Truncated int8_block payload")
        out = [0.0] * n
        for b in range(num_blocks):
            scale = scales[b]
            start = b * block_size
            end = min(n, start + block_size)
            if scale == 0.0:
                continue
            for i in range(start, end):
                qb = qbytes[i]
                signed_q = qb - 256 if qb >= 128 else qb
                out[i] = signed_q * scale
        return codec, out

    if codec == CODEC_INT4_BLOCK:
        num_blocks = (n + block_size - 1) // block_size
        scales = struct.unpack_from(f"<{num_blocks}f", payload, offset)
        offset += 4 * num_blocks
        packed = payload[offset : offset + (n + 1) // 2]
        if len(packed) != (n + 1) // 2:
            raise ValueError("Truncated int4_block payload")
        out = [0.0] * n
        for b in range(num_blocks):
            scale = scales[b]
            start = b * block_size
            end = min(n, start + block_size)
            if scale == 0.0:
                continue
            for i in range(start, end):
                byte_val = packed[i >> 1]
                nibble = (byte_val & 0x0F) if (i & 1) == 0 else ((byte_val >> 4) & 0x0F)
                signed_q = nibble - 7
                out[i] = signed_q * scale
        return codec, out

    if codec == CODEC_SIGN_1BIT_NO_EF:
        (mean_abs,) = struct.unpack_from("<f", payload, offset)
        offset += 4
        packed = payload[offset : offset + (n + 7) // 8]
        if len(packed) != (n + 7) // 8:
            raise ValueError("Truncated sign_1bit_no_ef payload")
        out = [0.0] * n
        for i in range(n):
            bit = (packed[i >> 3] >> (i & 7)) & 1
            out[i] = mean_abs if bit else -mean_abs
        return codec, out

    raise AssertionError("Unreachable")
