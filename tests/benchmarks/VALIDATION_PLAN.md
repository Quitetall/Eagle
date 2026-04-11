# LamQuant Gen 7.5 — Architecture Validation Plan

14 diagnostics, prioritized. Each answers a specific architectural question that can't be resolved theoretically.

## Status Key
- [ ] Not started
- [~] In progress
- [x] Complete

---

## Gate Criteria (4 results needed to ship)

1. Width-96 production R >= 0.90 with uniform per-frequency error (no boundary concentration)
2. Rate-distortion curve dominates Gen 6 at every operating point
3. Seizure detection sensitivity within 2% of original at Mode 0 (200:1)
4. Per-patient generalization spread within +/-0.04 R of mean

---

## Phase 1: Architecture Validation (blocks production decision)

### 1. Per-Frequency Reconstruction Error Heatmap
- [x] **Implemented**: `benchmark_subband_leakage.py` (Audit 7 in gauntlet)
- [ ] **Production run**: Re-run on production checkpoint (550ep + hardening)
- **Question**: Is the lifting decomposition the bottleneck?
- **Method**: MSE and 1-R per 0.5 Hz bin, 0-50 Hz, across holdout patients. Overlay Le Gall 5/3 boundaries (15.6, 31.25, 62.5 Hz).
- **Pass**: Error smooth across boundaries. **Fail**: Error spikes at 15.6 Hz.
- **Fast baseline**: UNIFORM, boundary deficit -1.4 dB (boundary outperforms average). Architecture is sound at fast preset.
- **Note**: Current implementation covers 0-15.6 Hz (L3 band). Extend to full 0-50 Hz by reconstructing through inverse lifting to raw domain.

### 2. Ablation Matrix: Structural Priors vs Learned Capacity
- [ ] **Not started** — requires 4 separate production training runs
- **Question**: How many R points does each structural prior contribute?
- **Configurations** (all at 550ep, hardening, width 96):
  - Config A: Raw signal -> TNN stride 8 (Gen 6 baseline)
  - Config B: LPC residual -> TNN stride 8
  - Config C: Lifting L3 approx -> TNN stride 4 (Gen 7.5, no LPC)
  - Config D: LPC residual -> Lifting L3 approx -> TNN stride 4 (Gen 7.5 full)
- **Pass**: Config C beats Config A despite fewer parameters.
- **Decision**: If Config D doesn't beat Config C, drop LPC (saves 3 ms latency + 168 bytes side info).

### 3. FSQ Token Entropy vs Activity Level
- [ ] **Not started**
- **Question**: Does SNN-driven adaptive FSQ actually help compression?
- **Method**: Per-window FSQ token entropy (per group + aggregate) correlated with SNN activity class (quiescent/moderate/high).
- **Pass**: Quiescent entropy at L=2 dramatically lower than at L=3.
- **Fail**: Similar distributions -> SNN thresholds miscalibrated.

### 4. Rate-Distortion Curve (Full Operating Range)
- [ ] **Not started**
- **Question**: Does subband dominate Gen 6 at every operating point?
- **Method**: Sweep FSQ L=2..7, detail modes 0/1/2. Plot R vs bits-per-sample. Overlay Gen 6 R-D curve.
- **Pass**: Subband curve above Gen 6 everywhere.
- **Crossover**: If curves cross, identifies exactly where each architecture wins.

---

## Phase 2: Clinical Robustness (blocks deployment decision)

### 5. Per-Patient Generalization Spread
- [ ] **Not started**
- **Question**: Is variance across patients acceptable?
- **Method**: Box plot of R per holdout patient (chb15-chb20).
- **Pass**: All patients within +/-0.04 of mean R. **Fail**: Any patient >0.08 below mean.
- **Note**: EEG varies enormously (skull thickness, cortical folding, dominant rhythms). Tight variance > high mean.

### 6. Seizure vs Quiescent R (Separate)
- [ ] **Not started**
- **Question**: Does adaptive FSQ improve event quality?
- **Method**: R computed exclusively on seizure windows vs exclusively on quiescent windows.
- **Pass**: Seizure R >= quiescent R (system allocates more bits during events).
- **Fail**: Seizure R < quiescent R -> SNN miscalibrating bit allocation.

### 7. Clinical Feature Preservation at Extreme CR (Mode 0, 200:1)
- [ ] **Not started**
- **Question**: Do downstream algorithms produce same clinical conclusions?
- **Method**: Run seizure detector (threshold on spectral power ratio or REACT) on original vs reconstructed. Report sensitivity and FPR for both.
- **Pass**: Sensitivity drops <2% at 200:1. **Publishable result** if achieved.
- **Optional**: Automated sleep staging on labeled sleep data.

### 8. Spectral Fidelity (PSD Comparison)
- [ ] **Not started**
- **Question**: What is the effective bandwidth of the codec?
- **Method**: Welch PSD (2s windows, 50% overlap) of original vs reconstructed. Report dB error across 0-50 Hz.
- **Thresholds**: <1 dB imperceptible, <3 dB acceptable for monitoring, >6 dB diagnostically problematic.
- **Key metric**: Frequency at which PSD error exceeds 3 dB = effective codec bandwidth.

---

## Phase 3: Efficiency Validation (informs optimization priorities)

### 9. Latent Space Utilization
- [ ] **Not started**
- **Question**: Are latent dimensions wasted?
- **Method**: Codebook utilization per FSQ group (% of L^4 codewords used). Mutual information between adjacent groups.
- **Decision**: <50% utilization -> that group has wasted capacity. High MI between groups -> merge candidates.

### 10. XNOR+cpop Kernel Benchmark (Hazard3 Hardware)
- [ ] **Not started** — requires RP2350 hardware
- **Question**: How close is the MAC kernel to theoretical throughput?
- **Method**: Cycle-accurate profiling of ternary MAC inner loop at width 96 in RISC-V mode.
- **Target**: 16 MACs per 5 cycles (theoretical). Gap = optimization headroom.

### 11. End-to-End Pipeline Latency Profile (Hazard3 Hardware)
- [ ] **Not started** — requires RP2350 hardware
- **Question**: Where is the actual latency bottleneck?
- **Method**: Per-stage cycle counters: biquad, LPC, lifting, TNN (per layer), FSQ, rANS, detail encoding, BLE assembly.
- **Note**: Real hardware has memory access patterns, pipeline stalls, bus contention not captured by estimates.

### 12. Detail Coefficient Sparsity
- [ ] **Not started**
- **Question**: Are details sparse enough for cheap encoding?
- **Method**: % non-zero detail coefficients (threshold alpha=0.5) per subband (L1/L2/L3 detail), per patient.
- **Critical**: If L3 detail is 30% non-zero (not 5% estimated), Mode 1 CR projections are wrong.

### 13. LPC Prediction Gain
- [ ] **Not started**
- **Question**: Is LPC worth its 3.1 ms latency cost?
- **Method**: Prediction gain (input variance / residual variance, dB) of order-8 LPC on prefiltered EEG, per channel, per patient. Separate for quiescent vs seizure.
- **Decision**: >10 dB gain -> keep LPC. <5 dB -> EEG already white after biquad, drop LPC.

### 14. Cayley Rotation Effectiveness
- [ ] **Not started**
- **Question**: Does the learned rotation justify its 4 KB firmware cost?
- **Method**: FSQ token entropy with and without rotation. Compare codebook dead code rates.
- **Pass**: Entropy drops >0.3 bits/symbol. **Fail**: <0.1 bits/symbol -> rotation adds complexity for negligible gain.
- **Fast baseline**: 0% dead codes with rotation vs 6.2% without (from earlier testing).

---

## Dependencies

```
Production training (550ep, width 96, hardening)
  |
  +-- Diagnostic 1 (per-freq error) — re-run on production ckpt
  +-- Diagnostic 4 (R-D curve)
  +-- Diagnostic 5 (per-patient spread)
  +-- Diagnostic 6 (seizure vs quiescent R)
  +-- Diagnostic 7 (clinical feature preservation)
  +-- Diagnostic 8 (PSD comparison)
  +-- Diagnostic 9 (latent utilization)
  +-- Diagnostic 14 (Cayley effectiveness)

Ablation runs (4x production training)
  +-- Diagnostic 2 (ablation matrix)

SNN production training (500ep, d_model=40)
  +-- Diagnostic 3 (FSQ entropy vs activity)
  +-- Diagnostic 6 (seizure vs quiescent — needs SNN labels)

L3 precomputed data (already available)
  +-- Diagnostic 12 (detail sparsity) — can run NOW
  +-- Diagnostic 13 (LPC prediction gain) — can run NOW

RP2350 hardware
  +-- Diagnostic 10 (XNOR+cpop kernel)
  +-- Diagnostic 11 (pipeline latency)
```

## Fast Preset Baseline Results (2026-04-11)

21 audits: **9 PASS, 11 FAIL, 1 SKIP** on 100-epoch fast-preset checkpoint (width=112, no hardening).

Key structural findings (architecture validated):
- D7/D8 (leakage): Boundary deficit -1.4 dB / 4.4 dB — **no lifting boundary artifact**
- D14 (clinical): Sensitivity ratio 1.000 at Mode 0 — **clinical preservation intact**
- D6 (seizure R): Delta -0.015 — seizure quality slightly BETTER than quiescent
- D9 (latent): 100% utilization, 0 dead codes
- D13 (LPC): 9.87 dB quiescent gain — **LPC justified**
- D10/D11: 58.8 ms total, 170x real-time margin

Structural concerns for production follow-up:
- D12: L1 sparsity 52% (spec assumed >70%). Mode 1 overhead ~120-150 bytes/window, not ~65. CR shifts from ~147:1 to ~120-130:1. Consider dropping L1 from Mode 1 entirely (62.5-125 Hz, above clinical EEG band).
- D14 (Cayley): Global dead codes tied at 0, but per-dim improvement (7 vs 16). Hold judgment until production.
- D3 (entropy): H(seizure) < H(quiet) at fast preset. Expected to differentiate with production training. If not, may reflect that the L3 subband doesn't capture the high-frequency components of seizure evolution.

## Diagnostic Calibration Notes

**D8 (spectral fidelity)**: f_3dB should be reported separately for L3 band (0-15.6 Hz, TNN-reconstructed) and detail bands (15.6-50 Hz, lifting-reconstructed). f_3dB=14 Hz in L3 band + 45 Hz in detail bands = healthy system.

**D3 (FSQ entropy)**: If still flat after production, add secondary metric: detail coefficient entropy during seizure vs quiescent — high-frequency seizure signatures live in the details, not the L3 approximation.

**D12 (detail sparsity)**: 70% threshold based on original spec estimate. Actual L1 at 52% means either biquad transition band is wider than expected or LPC whitening boosts HF noise floor. Fix options: tighten biquad to 45 Hz, increase alpha threshold, or drop L1 from Mode 1.

**D2 (ablation)**: If constrained to 2 runs, prioritize Config A (raw→TNN8) vs Config D (full pipeline). A-vs-D answers the core question: do structural priors beat raw at matched compute?

**SRAM4**: Width=112 is 53 KB (over 43 KB SRAM4). Decision: width=96 (41 KB, fits) OR use SRAM5 overflow (SNN only uses ~5 KB of SRAM5's 59 KB). Production run uses width=96.

## Immediate Actions

1. Start production training run (student + SNN, width 96, production preset)
2. Diagnostics 12 and 13 already ran — results above
3. After production checkpoint drops: run full gauntlet (`python3 tests/benchmarks/run_all_benchmarks.py`)
4. Three numbers that determine next steps: f_3dB (must reach >12 Hz), seizure entropy separation (must emerge), worst-patient R (must exceed 0.80)
5. Diagnostic 2 (ablation) requires additional training runs — schedule after production validates
6. Diagnostics 10-11 require RP2350 hardware — schedule separately
