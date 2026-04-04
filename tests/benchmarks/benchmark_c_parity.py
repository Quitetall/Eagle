#!/usr/bin/env python3
"""
LamQuant Gen 7 — C-Parity Benchmark
====================================
Phase 1: Bit-perfect static weight extraction (ternary unpacking)
Phase 2: Cascaded forward-pass drift through all encoder layers

Simulates BOTH the main conv path AND the shortcut path using
C-extracted weights. For strided blocks (focal2, focal3), the
shortcut is a strided TernaryConv1d — not identity — and must
be simulated with C weights to avoid false drift.

GroupNorm and ReLU use PyTorch (identical in float — these are
not ternary and don't introduce C vs Python drift).
"""
import torch
import torch.nn.functional as F
import os
import sys
import numpy as np
import re
from pathlib import Path


def find_project_root(marker='.git'):
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / marker).exists():
            return str(parent)
    raise RuntimeError("Project root not found")


ROOT_DIR = find_project_root()
sys.path.append(os.path.join(ROOT_DIR, 'ai_models', 'student'))
from train_ternary import TernaryMobileNetV5, TernaryConv1d

TERNARY_LUT = np.array([0, 1, -1, 0], dtype=np.int32)


def unpack_header_ternary(packed_bytes, shape):
    unpacked = []
    for b in packed_bytes:
        for j in range(4):
            val = (b >> (2 * j)) & 0x03
            unpacked.append(TERNARY_LUT[val])
    return np.array(unpacked[:np.prod(shape)]).reshape(shape)


def parse_clinical_header(path):
    if not os.path.exists(path):
        print(f"[!] FATAL: Header missing -> {path}")
        sys.exit(1)
    with open(path, 'r') as f:
        content = f.read()

    alphas = {}
    for m in re.finditer(
        r"const int32_t\s+([a-zA-Z0-9_]+)_alphas_q31\[.*?\]\s*.*?\s*=\s*\{(.*?)\};",
        content, re.DOTALL
    ):
        name = m.group(1)
        vals = re.findall(r"-?[0-9][0-9]*", m.group(2))
        alphas[name] = np.array([int(v) for v in vals], dtype=np.int32).astype(np.float32) / 2147483647.0

    weights = {}
    for m in re.finditer(
        r"const uint8_t\s+([a-zA-Z0-9_]+)_weights\[.*?\]\s*.*?\s*=\s*\{(.*?)\};",
        content, re.DOTALL
    ):
        name = m.group(1)
        hex_vals = re.findall(r"0x[0-9a-fA-F]+", m.group(2))
        weights[name] = np.array([int(h, 16) for h in hex_vals], dtype=np.uint8)

    return alphas, weights


def c_conv1d(x_t, unpacked_w, alphas, conv_module):
    """Simulate a ternary conv1d using C-extracted weights and Q31 alphas."""
    w = torch.from_numpy(unpacked_w).float()
    for oc in range(w.shape[0]):
        w[oc] *= alphas[oc]
    if isinstance(x_t, np.ndarray):
        x_t = torch.from_numpy(x_t).float()
    return F.conv1d(x_t, w, bias=conv_module.bias,
                    stride=conv_module.stride, padding=conv_module.padding,
                    dilation=conv_module.dilation, groups=conv_module.groups)


def c_simulate_focal_block(x_t, conv_name, shortcut_name, block,
                            c_unpack, c_alphas):
    """
    Simulate a full TernaryFocalBlock using C-extracted weights:
      out = ReLU(GroupNorm(C_conv(x))) + C_shortcut(x)

    GroupNorm and ReLU use PyTorch (identical math).
    Conv and shortcut use C-extracted ternary weights + Q31 alphas.
    """
    # Main path: C-simulated conv → PyTorch GroupNorm → ReLU
    c_conv_out = c_conv1d(x_t, c_unpack[conv_name], c_alphas[conv_name], block.conv)
    c_normed = F.relu(block.norm(c_conv_out))

    # Shortcut path
    if isinstance(block.shortcut, TernaryConv1d):
        # Strided ternary shortcut — simulate with C weights
        c_sc_out = c_conv1d(x_t, c_unpack[shortcut_name], c_alphas[shortcut_name],
                            block.shortcut)
    else:
        # Identity shortcut (same channels, stride=1)
        c_sc_out = x_t

    return c_normed + c_sc_out


def run():
    device = 'cpu'
    model = TernaryMobileNetV5(in_ch=21, latent_dim=32)
    s_path = os.path.join(ROOT_DIR, "ai_models/student/student_hardened.ckpt")
    try:
        model.load_state_dict(torch.load(s_path, map_location=device))
    except Exception as e:
        print(f"[!] FATAL: Could not load checkpoint -> {e}")
        sys.exit(1)
    model.eval()

    header_path = os.path.join(ROOT_DIR, 'firmware/focal_net_weights.h')
    c_alphas, c_packed = parse_clinical_header(header_path)

    # All encoder layers (conv + shortcut for each block, plus bottleneck)
    encoder_convs = [
        ('focal1_conv', model.focal1.conv),
        ('focal1_shortcut', model.focal1.shortcut),
        ('focal2_conv', model.focal2.conv),
        ('focal2_shortcut', model.focal2.shortcut),
        ('focal3_conv', model.focal3.conv),
        ('focal3_shortcut', model.focal3.shortcut),
        ('focal4_conv', model.focal4.conv),
        ('focal4_shortcut', model.focal4.shortcut),
        ('bottleneck', model.bottleneck),
    ]

    # ===== PHASE 1: Static weight parity =====
    print("[*] Phase 1: Verifying bit-perfect weight extraction...")
    c_unpack = {}

    for c_name, py_layer in encoder_convs:
        if not hasattr(py_layer, 'lsq_alpha'):
            continue  # Skip identity shortcuts
        if c_name not in c_packed:
            print(f"[!] FATAL: {c_name}_weights missing from header")
            sys.exit(1)

        shape = py_layer.weight.shape
        c_unpack[c_name] = unpack_header_ternary(c_packed[c_name], shape)

        with torch.no_grad():
            w = py_layer.weight.detach().numpy()
            alpha = torch.abs(py_layer.lsq_alpha).detach().numpy().flatten()
            w_q = np.zeros_like(w, dtype=np.int8)
            for oc in range(w.shape[0]):
                w_q[oc] = np.round(np.clip(w[oc] / (alpha[oc] + 1e-8), -1, 1))

            mismatches = np.sum(w_q != c_unpack[c_name])
            if mismatches != 0:
                print(f"[FAIL] Weight drift in {c_name}: {mismatches} mismatches")
                sys.exit(1)
            print(f"  {c_name}: {np.prod(shape)} weights OK")

    # ===== PHASE 2: Cascaded forward-pass drift =====
    print("[*] Phase 2: Cascaded forward-pass drift (all encoder layers)...")

    np.random.seed(42)
    test_inputs = [
        np.zeros((1, 21, 2500), dtype=np.float32),
        np.ones((1, 21, 2500), dtype=np.float32) * 50.0,
        np.ones((1, 21, 2500), dtype=np.float32) * -50.0,
    ]
    for _ in range(10):
        test_inputs.append(
            np.random.uniform(-50.0, 50.0, size=(1, 21, 2500)).astype(np.float32))

    max_drift = 0.0

    with torch.no_grad():
        for x_np in test_inputs:
            x_t = torch.from_numpy(x_np).float()

            # PyTorch reference
            py_f1 = model.focal1(x_t, quantize=True)
            py_f2 = model.focal2(py_f1, quantize=True)
            py_f3 = model.focal3(py_f2, quantize=True)
            py_f4 = model.focal4(py_f3, quantize=True)
            py_lat = model.bottleneck(py_f4, quantize=True)

            # C-simulation: both conv AND shortcut use C-extracted weights
            c_f1 = c_simulate_focal_block(
                x_t, 'focal1_conv', 'focal1_shortcut',
                model.focal1, c_unpack, c_alphas)
            d1 = np.max(np.abs(py_f1.numpy() - c_f1.numpy()))

            c_f2 = c_simulate_focal_block(
                c_f1, 'focal2_conv', 'focal2_shortcut',
                model.focal2, c_unpack, c_alphas)
            d2 = np.max(np.abs(py_f2.numpy() - c_f2.numpy()))

            c_f3 = c_simulate_focal_block(
                c_f2, 'focal3_conv', 'focal3_shortcut',
                model.focal3, c_unpack, c_alphas)
            d3 = np.max(np.abs(py_f3.numpy() - c_f3.numpy()))

            c_f4 = c_simulate_focal_block(
                c_f3, 'focal4_conv', 'focal4_shortcut',
                model.focal4, c_unpack, c_alphas)
            d4 = np.max(np.abs(py_f4.numpy() - c_f4.numpy()))

            # Bottleneck (no norm, no shortcut)
            c_lat = c_conv1d(c_f4, c_unpack['bottleneck'],
                             c_alphas['bottleneck'], model.bottleneck)
            d_lat = np.max(np.abs(py_lat.numpy() - c_lat.numpy()))

            layer_max = max(d1, d2, d3, d4, d_lat)
            if layer_max > max_drift:
                max_drift = layer_max

    print(f"[*] Max cascaded encoder drift: {max_drift:.6f}")
    if max_drift > 0.01:
        print(f"[!] FATAL: Cascaded drift {max_drift:.6f} > 0.01 threshold")
        sys.exit(1)

    print("[PASS] Benchmark Bit Parity - CLEAN.")
    return max_drift


if __name__ == "__main__":
    run()
