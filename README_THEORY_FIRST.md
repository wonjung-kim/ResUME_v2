# ResUME theory-first dual packet implementation

This repository is a theory-first revision of ResUME that supports **both** packetization models from the paper redesign:

- `--packet_mode stage`: one CRC-protected packet per RVQ stage. The receiver keeps the longest valid global stage prefix. This is the **stage-level packet theorem** implementation (`G=1`).
- `--packet_mode group`: every RVQ stage is partitioned into non-overlapping spatial groups. Each group has an independent CRC/prefix state. This is the **group-wise packet** implementation and uses the retained-MI lower-bound formulation.

The same `resume.py`, training code, entropy estimator, modulation-policy optimizer, and test code handle both modes. No CA-NSVQ Gaussian replacement is used in the main path; training uses the actual hard RVQ/QAM/CRC result with a straight-through gradient.

## Theory-to-code mapping

For RVQ indices `K_l`, define the nested stage prefix `U_l=(K_1,...,K_l)` and the incremental information

`Delta_l = I(Z; K_l | K_<l) = H(K_l | K_<l)`

for deterministic RVQ.

### Stage mode

With one packet per stage and longest-valid-prefix decoding,

`I(Z;Y) = sum_l Delta_l * P(T >= l)`.

Under conditionally independent packet errors,

`P(T >= l) = prod_{j<=l} s_j`,

so the policy objective is

`sum_l Delta_l * prod_{j<=l} s(M_j,SNR)`.

### Group mode

Let `A_{g,l}` be group `g`'s packet at stage `l`. A fixed stage-major packet ordering is used to define

`delta_{g,l} = H(A_{g,l} | all packets preceding A_{g,l})`.

For an arbitrary recovered packet subset, conditioning-reduces-entropy gives the retained-information lower bound

`I(Z;Y) >= sum_{g,l} delta_{g,l} P(T_g >= l)`.

All groups use the same modulation `M_l` at a given stage. Therefore the implemented lower-bound objective becomes

`sum_l (sum_g delta_{g,l}) * prod_{j<=l} s(M_j,SNR)`.

`stage` is exactly the `G=1` special case of the same packet code.

## Core files

- `resume.py`: RVQ, stage/group packetization, QAM, AWGN/Rayleigh channel, CRC validation, per-image/per-group prefix decoding, hard-path STE.
- `packet_utils.py`: index/bit conversion, zero padding, CRC-16, spatial group construction.
- `networks.py`: JSCC backbones and receiver-known prefix-depth conditioning.
- `train.py`: warm-up plus deployment-matched hard-prefix training for both modes.
- `entropy_model.py`: causal autoregressive entropy model using the same stage/group packet order as the theory.
- `estimate_information.py`: estimates stage increments in stage mode and packet/group increments in group mode.
- `modulation_policy.py`: packet success, CBR accounting, retained-information objective, exhaustive policy search.
- `build_policy.py`: builds SNR/CBR lookup tables from information estimates.
- `test.py`: Kodak evaluation using the generated policy.
- `sample_test.py`: quick image-folder evaluation with a fixed modulation profile.
- `smoke_test_dual.py`: dataset-free CPU smoke test for both packet modes.
- `pipeline_stage.sh`: complete stage-mode train -> information -> policy -> test pipeline.
- `pipeline_group.sh`: complete group-mode pipeline.

## Dataset paths

You can either use the original defaults or set paths explicitly.

```bash
export RESUME_IMAGENET_ROOT=/path/to/ImageNet   # contains train/ and val/
export RESUME_KODAK_ROOT=/path/to/Kodak         # contains Kodak *.png files
```

`train.py` and `estimate_information.py` also accept `--imagenet_root`; `test.py` accepts `--kodak_root`.

## 0. Install and smoke-test

```bash
pip install -r requirements.txt
python smoke_test_dual.py
```

The smoke test verifies:

1. stage packets execute end-to-end,
2. group packets execute end-to-end,
3. a one-group group configuration is numerically identical to stage mode under the same RNG state,
4. both packet layouts are accepted by the modulation-policy search.

---

# A. Stage-level theorem version

## A1. Train

```bash
python train.py \
  --model gauss \
  --stages 4 \
  --bits 12 \
  --batch 36 \
  --epochs 200 \
  --warmup_epochs 20 \
  --snr_min 0 \
  --snr_max 10 \
  --packet_mode stage \
  --norm \
  --out_dir ./output_stage/train
```

`stage` produces exactly one packet per active RVQ stage. For an 8x8 latent, each packet therefore contains all 64 RVQ indices of that stage plus CRC-16.

## A2. Estimate information increments

```bash
python estimate_information.py \
  --ckpt ./output_stage/train/best.pt \
  --model gauss \
  --stages 4 \
  --bits 12 \
  --packet_mode stage \
  --batch 8 \
  --epochs 10 \
  --norm \
  --out ./output_stage/information.json
```

The theory's MI decomposition is exact; the numerical `Delta_l` values are estimated using the learned causal entropy model.

## A3. Build the MI-optimal SNR/CBR policy

```bash
python build_policy.py \
  --info ./output_stage/information.json \
  --channel awgn \
  --snrs=-5,0,5,10,15,20 \
  --cbrs=0.00521,0.0104167,0.015625 \
  --out ./output_stage/policy.json
```

## A4. Test

```bash
python test.py \
  --ckpt ./output_stage/train/best.pt \
  --policy ./output_stage/policy.json \
  --channel awgn \
  --model gauss \
  --stages 4 \
  --bits 12 \
  --batch 24 \
  --norm \
  --out_dir ./output_stage/test
```

Or run the entire pipeline:

```bash
bash pipeline_stage.sh
```

---

# B. Group-wise packet version

For the default 8x8 latent, `--group_h 4 --group_w 4` gives 4 spatial groups, each containing 16 latent tokens. Every `(stage, group)` is an independent CRC-protected packet, while valid-prefix state is maintained independently per group.

## B1. Train

```bash
python train.py \
  --model gauss \
  --stages 4 \
  --bits 12 \
  --batch 36 \
  --epochs 200 \
  --warmup_epochs 20 \
  --snr_min 0 \
  --snr_max 10 \
  --packet_mode group \
  --group_h 4 \
  --group_w 4 \
  --norm \
  --out_dir ./output_group_4x4/train
```

Useful 8x8-latent group choices are:

- `2x2`: `G=16`, 4 tokens/group; shortest packets, largest CRC overhead.
- `2x4`: `G=8`, 8 tokens/group.
- `4x4`: `G=4`, 16 tokens/group; recommended first setting.
- `8x8`: `G=1`, identical packet partition to stage mode.

## B2. Estimate group-wise information increments

```bash
python estimate_information.py \
  --ckpt ./output_group_4x4/train/best.pt \
  --model gauss \
  --stages 4 \
  --bits 12 \
  --packet_mode group \
  --group_h 4 \
  --group_w 4 \
  --batch 8 \
  --epochs 10 \
  --norm \
  --out ./output_group_4x4/information.json
```

The JSON contains both:

- `delta_packet_bits`: `[L,G]`, the estimated `delta_{g,l}` values,
- `delta_stage_bits`: the sum over groups used by the common-stage-modulation policy.

## B3. Build the lower-bound-optimal policy

```bash
python build_policy.py \
  --info ./output_group_4x4/information.json \
  --channel awgn \
  --snrs=-5,0,5,10,15,20 \
  --cbrs=0.00521,0.0104167,0.015625 \
  --out ./output_group_4x4/policy.json
```

The group count and group size are read from `information.json`, so CBR includes the additional CRC overhead automatically.

## B4. Test

```bash
python test.py \
  --ckpt ./output_group_4x4/train/best.pt \
  --policy ./output_group_4x4/policy.json \
  --channel awgn \
  --model gauss \
  --stages 4 \
  --bits 12 \
  --batch 24 \
  --norm \
  --out_dir ./output_group_4x4/test
```

Or run the complete pipeline:

```bash
GROUP_H=4 GROUP_W=4 bash pipeline_group.sh
```

---

# C. Rayleigh block fading

Training:

```bash
python train.py ... --fading
```

Policy:

```bash
python build_policy.py ... --channel rayleigh
```

Testing:

```bash
python test.py ... --channel rayleigh
```

The implementation draws one independent flat-fading coefficient per CRC packet. Consequently, group mode has independent fading coefficients across group packets, consistent with the packet-level block-fading reliability used by `modulation_policy.py`.

# D. What is shared and what is different?

Both modes share the same learned RVQ architecture and the same QAM/channel implementation. Their only fundamental packet-level difference is the partition and prefix state:

- stage: `prefix_len` has shape `[B,1]`, global image prefix.
- group: `prefix_len` has shape `[B,G]`, independent spatial-group prefixes.

The decoder can use `depth_map`, which expands the receiver-known prefix length to every latent spatial token. Disable it with `--no-use_prefix_map` for an ablation.

# E. CBR and CRC

Every packet contains `num_indices * bits_per_index + 16` transmitted bits before QAM padding. CBR is computed from the exact number of QAM channel symbols after per-packet padding. Thus group mode correctly pays more CRC/padding overhead as the number of groups increases.

The analytic packet-success model treats a packet as successful when all modulation symbols are correctly detected. The implementation uses CRC-16 as the practical error detector; the theory therefore corresponds to the usual negligible-undetected-error approximation for CRC.

# F. Notes for the paper

- Do not restore the old claim `I(Z;Q_1) >= ... >= I(Z;Q_L)`.
- Stage mode uses the exact nested-prefix MI formulation.
- Group mode should be described using the exact recovered-set entropy expression plus the implemented tractable lower bound.
- The theorem should be stated as a **reliability-ordering** theorem. Ascending QAM order follows only when the selected modulation family has the required reliability ordering at the operating SNR/packet length.
- `build_policy.py --all_permutations` is included to empirically check the theorem-supported ascending restriction against all candidate permutations.
