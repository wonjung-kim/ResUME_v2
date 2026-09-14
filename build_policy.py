import argparse
import json
import os

from modulation_policy import MOD_ORDERS, build_policy_lut, save_policy


def parse_floats(s):
    return [float(x) for x in s.split(",") if x]


def parse_ints(s):
    return [int(x) for x in s.split(",") if x]


def main():
    p = argparse.ArgumentParser("Build theory-first SNR/CBR modulation LUT for stage or group packets")
    p.add_argument("--info", required=True, help="JSON from estimate_information.py")
    p.add_argument("--channel", choices=["awgn", "rayleigh"], default="awgn")
    p.add_argument("--snrs", type=str, default="-5,0,5,10,15,20")
    p.add_argument("--cbrs", type=str, default="0.00521,0.0104167,0.015625")
    p.add_argument("--mod_orders", type=str, default="2,4,16,64,256")
    p.add_argument("--image_h", type=int, default=128)
    p.add_argument("--image_w", type=int, default=128)
    p.add_argument("--crc_bits", type=int, default=16)
    p.add_argument("--all_permutations", action="store_true",
                   help="Search all modulation permutations; default uses theorem-supported ascending profiles only")
    p.add_argument("--out", type=str, default="policy.json")
    args = p.parse_args()

    with open(args.info) as f:
        info = json.load(f)
    delta = info.get("delta_stage_bits", info["delta_bits_per_image"])
    packet_mode = info.get("packet_mode", "stage")
    num_groups = int(info.get("num_groups", 1))
    group_size = int(info.get("group_size", info["num_tokens"]))
    bits = int(info.get("bits_per_index", 12))
    snrs = parse_floats(args.snrs)
    cbrs = parse_floats(args.cbrs)
    mods = parse_ints(args.mod_orders)

    lut = build_policy_lut(
        delta,
        snrs,
        cbrs,
        num_groups,
        group_size,
        bits,
        args.image_h,
        args.image_w,
        channel=args.channel,
        crc_bits=args.crc_bits,
        mod_orders=mods,
        ascending_only=not args.all_permutations,
    )
    objective_type = info.get(
        "objective_type",
        "stage_exact_prefix_mi" if packet_mode == "stage" else "group_retained_mi_lower_bound",
    )
    metadata = {
        "channel": args.channel,
        "packet_mode": packet_mode,
        "objective_type": objective_type,
        "delta_stage_bits": delta,
        "delta_packet_bits": info.get("delta_packet_bits"),
        "num_tokens": int(info["num_tokens"]),
        "num_groups": num_groups,
        "group_size": group_size,
        "group_h": int(info.get("group_h", info.get("latent_h", 1))),
        "group_w": int(info.get("group_w", info.get("latent_w", group_size))),
        "latent_h": int(info.get("latent_h", 0)),
        "latent_w": int(info.get("latent_w", 0)),
        "bits_per_index": bits,
        "image_h": args.image_h,
        "image_w": args.image_w,
        "crc_bits": args.crc_bits,
        "mod_orders": mods,
        "ascending_only": not args.all_permutations,
        "theory": (
            "stage: exact sum_l Delta_l prod_{j<=l}s_j; "
            "group: lower bound sum_l(sum_g delta_g_l) prod_{j<=l}s_j; optimize under CBR"
        ),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_policy(args.out, metadata, lut)
    print(f"Saved {args.out} ({packet_mode}, G={num_groups}, group_size={group_size})")


if __name__ == "__main__":
    main()
