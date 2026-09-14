import math
from functools import lru_cache
from typing import Tuple

import torch
import torch.nn.functional as F


def int_to_bits(values: torch.Tensor, width: int) -> torch.Tensor:
    """Convert non-negative integers [...] to bits [..., width] (MSB first)."""
    shifts = torch.arange(width - 1, -1, -1, device=values.device, dtype=torch.long)
    return ((values.long().unsqueeze(-1) >> shifts) & 1).to(torch.uint8)


def bits_to_int(bits: torch.Tensor) -> torch.Tensor:
    """Convert bits [..., width] (MSB first) to integer tensor [...]."""
    width = bits.shape[-1]
    powers = 2 ** torch.arange(width - 1, -1, -1, device=bits.device, dtype=torch.long)
    return (bits.long() * powers).sum(dim=-1)


def indices_to_payload_bits(indices: torch.Tensor, bits_per_index: int) -> torch.Tensor:
    """indices [B,N] -> payload [B,N*bits_per_index]."""
    return int_to_bits(indices, bits_per_index).reshape(indices.shape[0], -1)


def payload_bits_to_indices(payload: torch.Tensor, bits_per_index: int, num_indices: int) -> torch.Tensor:
    """payload [B,>=N*b] -> indices [B,N]."""
    need = num_indices * bits_per_index
    payload = payload[:, :need]
    return bits_to_int(payload.reshape(payload.shape[0], num_indices, bits_per_index))


def _make_crc16_ccitt_table(poly: int = 0x1021):
    table = []
    for byte in range(256):
        crc = byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
        table.append(crc)
    return table


_CRC16_TABLE_CPU = _make_crc16_ccitt_table()
_CRC16_DEVICE_CACHE = {}


def _crc_table(device: torch.device) -> torch.Tensor:
    key = str(device)
    if key not in _CRC16_DEVICE_CACHE:
        _CRC16_DEVICE_CACHE[key] = torch.tensor(_CRC16_TABLE_CPU, dtype=torch.long, device=device)
    return _CRC16_DEVICE_CACHE[key]


def _bits_to_bytes(bits: torch.Tensor) -> torch.Tensor:
    """[B,P] binary bits -> [B,ceil(P/8)] uint-like longs; final partial byte is zero-padded."""
    if bits.ndim != 2:
        raise ValueError("bits must have shape [B,P]")
    pad = (-bits.shape[1]) % 8
    if pad:
        bits = F.pad(bits, (0, pad), value=0)
    grouped = bits.reshape(bits.shape[0], -1, 8)
    return bits_to_int(grouped)


def crc16_ccitt(bits: torch.Tensor, init: int = 0xFFFF) -> torch.Tensor:
    """
    Batched CRC-16/CCITT-FALSE-like checksum over [B,P] bits.

    The payload is zero-padded to the next byte only for CRC computation when P is not
    byte-aligned. Both transmitter and receiver use the same convention. The byte-wise
    table implementation is substantially faster than a Python loop over every bit.
    """
    if bits.ndim != 2:
        raise ValueError("crc16_ccitt expects [B,P] bit tensor")
    byte_values = _bits_to_bytes(bits)
    crc = torch.full((bits.shape[0],), init, dtype=torch.long, device=bits.device)
    table = _crc_table(bits.device)
    for i in range(byte_values.shape[1]):
        lookup = ((crc >> 8) ^ byte_values[:, i].long()) & 0xFF
        crc = (((crc << 8) & 0xFFFF) ^ table[lookup]) & 0xFFFF
    return int_to_bits(crc, 16)


def append_crc16(payload: torch.Tensor) -> torch.Tensor:
    return torch.cat([payload, crc16_ccitt(payload)], dim=1)


def split_and_check_crc16(packet_bits: torch.Tensor, payload_bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return payload [B,P] and CRC validity [B]."""
    payload = packet_bits[:, :payload_bits]
    received_crc = packet_bits[:, payload_bits:payload_bits + 16]
    expected_crc = crc16_ccitt(payload)
    valid = (received_crc == expected_crc).all(dim=1)
    return payload, valid


def pad_bits(bits: torch.Tensor, bits_per_symbol: int) -> Tuple[torch.Tensor, int]:
    pad = (bits_per_symbol - (bits.shape[1] % bits_per_symbol)) % bits_per_symbol
    if pad:
        bits = F.pad(bits, (0, pad), value=0)
    return bits, pad


def bits_to_symbols(bits: torch.Tensor, bits_per_symbol: int) -> Tuple[torch.Tensor, int]:
    """Binary bits [B,P] -> natural-labelled symbol indices [B,S]."""
    bits, pad = pad_bits(bits, bits_per_symbol)
    grouped = bits.reshape(bits.shape[0], -1, bits_per_symbol)
    return bits_to_int(grouped), pad


def symbols_to_bits(symbol_indices: torch.Tensor, bits_per_symbol: int, original_bit_count: int) -> torch.Tensor:
    bits = int_to_bits(symbol_indices, bits_per_symbol).reshape(symbol_indices.shape[0], -1)
    return bits[:, :original_bit_count]


def packet_symbol_count(
    num_indices: int,
    bits_per_index: int,
    modulation_order: int,
    crc_bits: int = 16,
) -> int:
    if crc_bits != 16:
        raise ValueError("The current packet implementation uses CRC-16, so crc_bits must be 16")
    payload = num_indices * bits_per_index + crc_bits
    bps = int(round(math.log2(modulation_order)))
    return math.ceil(payload / bps)


def build_spatial_group_indices(
    latent_h: int,
    latent_w: int,
    group_h: int,
    group_w: int,
    device=None,
) -> torch.Tensor:
    """
    Return [G,m] flattened token indices for non-overlapping rectangular spatial groups.

    For an 8x8 latent and group_h=group_w=4, returns G=4 groups with m=16 tokens each.
    """
    if group_h <= 0 or group_w <= 0:
        raise ValueError("group_h and group_w must be positive")
    if latent_h % group_h != 0 or latent_w % group_w != 0:
        raise ValueError(
            f"latent ({latent_h},{latent_w}) must be divisible by group ({group_h},{group_w})"
        )
    groups = []
    for r0 in range(0, latent_h, group_h):
        for c0 in range(0, latent_w, group_w):
            ids = []
            for r in range(r0, r0 + group_h):
                for c in range(c0, c0 + group_w):
                    ids.append(r * latent_w + c)
            groups.append(ids)
    return torch.tensor(groups, dtype=torch.long, device=device)


def packet_group_indices(
    packet_mode: str,
    latent_h: int,
    latent_w: int,
    group_h: int = 4,
    group_w: int = 4,
    device=None,
) -> torch.Tensor:
    """Unified packet partition: stage mode is the special case G=1 containing all tokens."""
    N = latent_h * latent_w
    if packet_mode == "stage":
        return torch.arange(N, dtype=torch.long, device=device).reshape(1, N)
    if packet_mode == "group":
        return build_spatial_group_indices(latent_h, latent_w, group_h, group_w, device=device)
    raise ValueError("packet_mode must be 'stage' or 'group'")
