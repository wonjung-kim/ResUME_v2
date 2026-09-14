import itertools
import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np


MOD_ORDERS = (2, 4, 16, 64, 256)


def qfunc(x: float | np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * np.vectorize(math.erfc)(x / math.sqrt(2.0))


def symbol_error_awgn(M: int, snr_linear: float | np.ndarray):
    """Uncoded BPSK / square-QAM symbol error probability at unit average symbol power."""
    g = np.asarray(snr_linear, dtype=np.float64)
    if M == 2:
        return qfunc(np.sqrt(2.0 * g))
    m = int(round(math.sqrt(M)))
    if m * m != M:
        raise ValueError("Only BPSK and square QAM are supported")
    a = 2.0 * (1.0 - 1.0 / m) * qfunc(np.sqrt(3.0 * g / (M - 1.0)))
    return 1.0 - (1.0 - a) ** 2


def _all_correct_prob(symbol_error, num_symbols: int):
    ps = np.clip(symbol_error, 0.0, 1.0 - 1e-15)
    return np.exp(num_symbols * np.log1p(-ps))


def packet_success_awgn(M: int, snr_db: float, num_symbols: int) -> float:
    gamma = 10.0 ** (snr_db / 10.0)
    return float(_all_correct_prob(symbol_error_awgn(M, gamma), num_symbols))


def packet_success_rayleigh(M: int, snr_db: float, num_symbols: int, quadrature_order: int = 64) -> float:
    """
    E_h[(1-Ps(M,gamma*|h|^2))^n] for |h|^2~Exp(1), using Gauss-Laguerre quadrature.
    This matches the implementation's independent block fade per packet.
    """
    gamma = 10.0 ** (snr_db / 10.0)
    x, w = np.polynomial.laguerre.laggauss(quadrature_order)
    p = _all_correct_prob(symbol_error_awgn(M, gamma * x), num_symbols)
    return float(np.sum(w * p))


def packet_symbol_count(num_indices: int, bits_per_index: int, M: int, crc_bits: int = 16) -> int:
    packet_bits = num_indices * bits_per_index + crc_bits
    return math.ceil(packet_bits / math.log2(M))


def cbr_for_profile(
    profile: Sequence[int],
    num_groups: int,
    group_size: int,
    bits_per_index: int,
    image_h: int,
    image_w: int,
    crc_bits: int = 16,
) -> float:
    symbols = 0
    for M in profile:
        symbols += num_groups * packet_symbol_count(group_size, bits_per_index, M, crc_bits)
    return symbols / (3.0 * image_h * image_w)


def retained_information_objective(
    delta_stage_bits: Sequence[float],
    profile: Sequence[int],
    snr_db: float,
    group_size: int,
    bits_per_index: int,
    channel: str = "awgn",
    crc_bits: int = 16,
) -> float:
    """
    stage mode: exact retained-prefix MI objective when delta estimates are exact.
    group mode: rigorous lower-bound objective after summing packet increments over groups.

    All groups use the same M_l at stage l, so they share the same packet success s_l.
    """
    prefix_success = 1.0
    J = 0.0
    for l, M in enumerate(profile):
        nsym = packet_symbol_count(group_size, bits_per_index, M, crc_bits)
        if channel == "awgn":
            s = packet_success_awgn(M, snr_db, nsym)
        elif channel == "rayleigh":
            s = packet_success_rayleigh(M, snr_db, nsym)
        else:
            raise ValueError("channel must be 'awgn' or 'rayleigh'")
        prefix_success *= s
        J += float(delta_stage_bits[l]) * prefix_success
    return J


@dataclass
class PolicyEntry:
    active_stages: int
    mod_orders: Tuple[int, ...]
    cbr: float
    retained_information_bits: float

    @property
    def retained_mi_bits(self):
        # compatibility with first theory-first revision
        return self.retained_information_bits


def search_best_profile(
    delta_stage_bits: Sequence[float],
    snr_db: float,
    target_cbr: float,
    num_groups: int,
    group_size: int,
    bits_per_index: int,
    image_h: int,
    image_w: int,
    channel: str = "awgn",
    mod_orders: Sequence[int] = MOD_ORDERS,
    crc_bits: int = 16,
    ascending_only: bool = True,
) -> PolicyEntry:
    best = None
    Lmax = len(delta_stage_bits)
    min_cbr_seen = float("inf")
    for L in range(1, Lmax + 1):
        iterator = (
            itertools.combinations_with_replacement(mod_orders, L)
            if ascending_only
            else itertools.product(mod_orders, repeat=L)
        )
        for profile in iterator:
            cbr = cbr_for_profile(
                profile, num_groups, group_size, bits_per_index, image_h, image_w, crc_bits
            )
            min_cbr_seen = min(min_cbr_seen, cbr)
            if cbr > target_cbr + 1e-15:
                continue
            J = retained_information_objective(
                delta_stage_bits, profile, snr_db, group_size, bits_per_index, channel, crc_bits
            )
            if best is None or J > best.retained_information_bits:
                best = PolicyEntry(L, tuple(int(x) for x in profile), cbr, J)
    if best is None:
        raise RuntimeError(
            f"No feasible modulation profile under target CBR={target_cbr:g}; "
            f"minimum searched CBR was {min_cbr_seen:g}. Increase target CBR or modulation set."
        )
    return best


def build_policy_lut(
    delta_stage_bits: Sequence[float],
    snr_list: Iterable[float],
    cbr_list: Iterable[float],
    num_groups: int,
    group_size: int,
    bits_per_index: int,
    image_h: int,
    image_w: int,
    channel: str = "awgn",
    crc_bits: int = 16,
    mod_orders: Sequence[int] = MOD_ORDERS,
    ascending_only: bool = True,
) -> Dict[str, dict]:
    lut = {}
    for snr in snr_list:
        for cbr in cbr_list:
            entry = search_best_profile(
                delta_stage_bits,
                snr,
                cbr,
                num_groups,
                group_size,
                bits_per_index,
                image_h,
                image_w,
                channel=channel,
                mod_orders=mod_orders,
                crc_bits=crc_bits,
                ascending_only=ascending_only,
            )
            d = asdict(entry)
            d["retained_mi_bits"] = d["retained_information_bits"]
            lut[f"snr={float(snr):g}|cbr={float(cbr):.10g}"] = d
    return lut


def save_policy(path: str, metadata: dict, lut: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "lut": lut}, f, indent=2)


def load_policy(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def lookup_policy(policy: dict, snr_db: float, target_cbr: float):
    lut = policy["lut"]
    exact = f"snr={float(snr_db):g}|cbr={float(target_cbr):.10g}"
    if exact in lut:
        return lut[exact]
    best_key = None
    best_d = float("inf")
    for key in lut:
        a, b = key.split("|")
        s = float(a.split("=")[1])
        c = float(b.split("=")[1])
        d = abs(s - snr_db) + 1000.0 * abs(c - target_cbr)
        if d < best_d:
            best_d, best_key = d, key
    return lut[best_key]
