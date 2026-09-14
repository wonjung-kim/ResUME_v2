from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from packet_utils import packet_group_indices


@dataclass
class PacketLayout:
    packet_mode: str
    num_stages: int
    latent_h: int
    latent_w: int
    num_groups: int
    group_size: int
    group_h: int
    group_w: int
    stage_ids: torch.Tensor       # [S]
    token_ids: torch.Tensor       # [S]
    packet_ids: torch.Tensor      # [S]
    packet_stages: List[int]      # [P]
    packet_groups: List[int]      # [P]

    @property
    def seq_len(self) -> int:
        return int(self.stage_ids.numel())

    @property
    def num_packets(self) -> int:
        return len(self.packet_stages)


def build_packet_layout(
    num_stages: int,
    latent_h: int,
    latent_w: int,
    packet_mode: str,
    group_hw: Tuple[int, int] = (4, 4),
    device=None,
) -> PacketLayout:
    """
    Fixed stage-major packet order used for the information decomposition.

    stage: [stage1(all tokens)], [stage2(all tokens)], ...
    group: [stage1/group1], ... [stage1/groupG], [stage2/group1], ...

    For group mode, delta_{g,l}=H(A_{g,l}|all packets preceding it in this fixed order).
    """
    gh, gw = int(group_hw[0]), int(group_hw[1])
    groups = packet_group_indices(packet_mode, latent_h, latent_w, gh, gw, device=device)
    G, m = groups.shape
    stage_ids, token_ids, packet_ids = [], [], []
    packet_stages, packet_groups = [], []
    p = 0
    for l in range(num_stages):
        for g in range(G):
            ids = groups[g].tolist()
            stage_ids.extend([l] * len(ids))
            token_ids.extend(ids)
            packet_ids.extend([p] * len(ids))
            packet_stages.append(l)
            packet_groups.append(g)
            p += 1
    return PacketLayout(
        packet_mode=packet_mode,
        num_stages=num_stages,
        latent_h=latent_h,
        latent_w=latent_w,
        num_groups=int(G),
        group_size=int(m),
        group_h=gh if packet_mode == "group" else latent_h,
        group_w=gw if packet_mode == "group" else latent_w,
        stage_ids=torch.tensor(stage_ids, dtype=torch.long, device=device),
        token_ids=torch.tensor(token_ids, dtype=torch.long, device=device),
        packet_ids=torch.tensor(packet_ids, dtype=torch.long, device=device),
        packet_stages=packet_stages,
        packet_groups=packet_groups,
    )


def indices_to_sequence(idx_list: List[torch.Tensor], layout: PacketLayout) -> torch.Tensor:
    """idx_list: L tensors [B,N] -> target sequence [B,S] in the layout order."""
    stack = torch.stack(idx_list[: layout.num_stages], dim=1)  # [B,L,N]
    return stack[:, layout.stage_ids, layout.token_ids]


class CausalIndexEntropyModel(nn.Module):
    """
    Autoregressive estimator of the joint RVQ index distribution.

    Summing token NLLs within one packet estimates H(A_i | A_<i) under the fixed
    packet ordering. This supports the exact stage-level decomposition (G=1) and the
    rigorous group-wise retained-information lower bound.
    """

    def __init__(
        self,
        num_codes: int,
        num_stages: int,
        num_tokens: int,
        max_seq_len: int,
        d_model: int = 192,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_codes = int(num_codes)
        self.index_embed = nn.Embedding(num_codes, d_model)
        self.stage_embed = nn.Embedding(num_stages, d_model)
        self.spatial_embed = nn.Embedding(num_tokens, d_model)
        self.position_embed = nn.Embedding(max_seq_len, d_model)
        self.bos = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, num_codes)
        nn.init.normal_(self.bos, std=0.02)

    def forward(self, targets: torch.Tensor, stage_ids: torch.Tensor, token_ids: torch.Tensor):
        B, S = targets.shape
        if S > self.position_embed.num_embeddings:
            raise ValueError("Sequence longer than max_seq_len configured in entropy model")
        prev = torch.empty((B, S, self.index_embed.embedding_dim), device=targets.device)
        prev[:, :1] = self.bos.to(dtype=prev.dtype)
        if S > 1:
            prev[:, 1:] = self.index_embed(targets[:, :-1])
        pos = torch.arange(S, device=targets.device)
        x = (
            prev
            + self.stage_embed(stage_ids).unsqueeze(0)
            + self.spatial_embed(token_ids).unsqueeze(0)
            + self.position_embed(pos).unsqueeze(0)
        )
        causal_mask = torch.triu(torch.ones(S, S, device=targets.device, dtype=torch.bool), diagonal=1)
        h = self.transformer(x, mask=causal_mask)
        return self.out(self.norm(h))


def token_nll_bits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-token cross entropy in bits, shape [B,S]."""
    B, S, K = logits.shape
    nll = F.cross_entropy(logits.reshape(B * S, K), targets.reshape(B * S), reduction="none")
    return (nll / torch.log(torch.tensor(2.0, device=logits.device))).reshape(B, S)


def packet_nll_per_sample(nll_bits: torch.Tensor, layout: PacketLayout) -> torch.Tensor:
    """Sum token NLLs inside each packet, returning [B,P] bits/packet."""
    B = nll_bits.shape[0]
    out = nll_bits.new_zeros((B, layout.num_packets))
    ids = layout.packet_ids.unsqueeze(0).expand(B, -1)
    out.scatter_add_(1, ids, nll_bits)
    return out


def packet_matrix(packet_values: torch.Tensor, layout: PacketLayout) -> torch.Tensor:
    """Convert [P] packet values to [L,G] according to the layout metadata."""
    out = packet_values.new_zeros((layout.num_stages, layout.num_groups))
    for p, (l, g) in enumerate(zip(layout.packet_stages, layout.packet_groups)):
        out[l, g] = packet_values[p]
    return out
