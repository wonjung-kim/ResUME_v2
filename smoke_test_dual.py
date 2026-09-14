"""Fast CPU smoke tests for the dual packet implementation; no dataset is required."""
import torch
from resume import ResUME
from entropy_model import build_packet_layout, indices_to_sequence
from modulation_policy import cbr_for_profile, search_best_profile


def run_resume(mode, group_hw):
    torch.manual_seed(0)
    model = ResUME(num_stages=3, vq_bitrate_per_stage=3, data_dim=4, device="cpu")
    z = torch.randn(2, 16, 4)
    model.train()
    out = model(
        z,
        active_stages=3,
        snr_db=torch.tensor([50.0, 50.0]),
        mod_orders=[2, 4, 16],
        packet_mode=mode,
        latent_hw=(4, 4),
        group_hw=group_hw,
    )
    assert out["z_out"].shape == z.shape
    expected_g = 1 if mode == "stage" else 4
    assert out["prefix_len"].shape == (2, expected_g)
    assert torch.all(out["prefix_len"] == 3)
    print(f"[{mode}] G={out['num_groups']}, group_size={out['group_size']}, symbols={out['symbol_counts']}")


def run_equivalence():
    torch.manual_seed(7)
    model = ResUME(2, 3, 4, device="cpu")
    z = torch.randn(2, 16, 4)
    model.eval()
    torch.manual_seed(123)
    stage = model(z, 2, torch.tensor([5.0, 5.0]), [4, 16], packet_mode="stage", latent_hw=(4, 4))
    torch.manual_seed(123)
    group_one = model(
        z, 2, torch.tensor([5.0, 5.0]), [4, 16], packet_mode="group", latent_hw=(4, 4), group_hw=(4, 4)
    )
    assert torch.allclose(stage["z_hard"], group_one["z_hard"])
    assert torch.equal(stage["prefix_len"], group_one["prefix_len"])
    print("[equivalence] stage mode == group mode with G=1")


def run_layout_policy():
    for mode, ghw in [("stage", (4, 4)), ("group", (2, 2))]:
        layout = build_packet_layout(3, 4, 4, mode, ghw, device="cpu")
        idx = [torch.arange(16).reshape(1, 16) % 8 for _ in range(3)]
        seq = indices_to_sequence(idx, layout)
        assert seq.shape == (1, 48)
        target = max(
            0.05,
            4.0 * cbr_for_profile([256], layout.num_groups, layout.group_size, 3, 128, 128),
        )
        best = search_best_profile(
            [12.0, 8.0, 4.0], 5.0, target,
            layout.num_groups, layout.group_size, 3, 128, 128,
        )
        print(f"[{mode}] layout P={layout.num_packets}, best={best.mod_orders}, CBR={best.cbr:.6g}")


if __name__ == "__main__":
    run_resume("stage", (4, 4))
    run_resume("group", (2, 2))
    run_equivalence()
    run_layout_policy()
    print("All dual-packet smoke tests passed.")
