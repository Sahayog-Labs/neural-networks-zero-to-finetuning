#!/usr/bin/env python3
"""
rnn_bottleneck.py -- the sequential bottleneck that forced attention, timed for real.

Course artifact for p.28 ("RNNs and the Sequential Bottleneck: Why Attention Had to
Exist"). Times two ways of doing the SAME total floating-point work on a real
torch tensor:

  1. THE RNN PATH: 2,048 sequential matrix-vector products, h_i = W @ h_{i-1}.
     Step i cannot start until step i-1's OUTPUT exists -- a genuine data
     dependency (not just a Python for-loop pretending to be one), because each
     iteration feeds the previous iteration's tensor back in.
  2. THE TRANSFORMER PATH: one (2048x4096) @ (4096x4096) batched matmul --
     2,048 INDEPENDENT matrix-vector products against the same weight matrix,
     issued as a single kernel launch, with IDENTICAL total FLOPs by construction
     (both do 2,048 x (2 x 4096^2) multiply-adds against the same W).

The page's point (brief S4, do not miss it): this is NOT a FLOPs race -- both
paths do the same arithmetic. The RNN loses because of DEPENDENCY, not ARITHMETIC:
step i must wait for step i-1's result before it can even be launched, so the
2,048 steps cannot overlap on the GPU (or here, cannot pipeline in the CPU's
BLAS calls). The batched matmul has no such constraint and one big kernel beats
2,048 tiny ones even though it is doing 2,048x the work of any single one of them.

FLOP/byte accounting (constants.md S6.5, d_h=4096, bf16 weights = 2 bytes/param,
matching the page's frozen worked example exactly):
  FLOPs/step = 2 * d_h^2 = 33.55 MFLOP        bytes/step = d_h^2 * 2 = 33.55 MB
  I_RNN         = 33.55 MFLOP / 33.55 MB = 1.0 FLOP/byte      (one step, reads W once)
  I_transformer = (2048 * 33.55 MFLOP) / 33.55 MB = 2048 FLOP/byte  (W reused 2048x, one read)

The wall-clock timing itself runs in fp32 (bf16 GEMM/GEMV is not consistently
BLAS-accelerated on CPU across platforms, and this script is meant to finish in
seconds on a laptop with no GPU, per spec-code.md D.6). The FLOP/byte accounting
above is reported at the page's bf16 byte count regardless of the compute dtype
used to produce the wall-clock numbers -- intensity is a property of the weight
matrix's storage format, not of what dtype happened to run the demo.

SAFETY: CPU only (no CUDA required), a few seconds, allocates ~67 MB (two 4096x4096
fp32 weight copies + working vectors). Writes and installs nothing.
"""

import time

import torch

SEED = 42
torch.manual_seed(SEED)

print("=" * 72)
print("rnn_bottleneck.py -- the sequential bottleneck that forced attention")
print(f"torch {torch.__version__} - seed {SEED}")
print("=" * 72)

# --------------------------------------------------------------------------
# The page's frozen worked example (p.28): d_h=4096 (Qwen3-8B residual width,
# constants.md S1.1), S=2048, weights in bf16 (2 bytes/param).
# --------------------------------------------------------------------------
D_H = 4096
S = 2048
BF16_BYTES = 2

flops_per_step = 2 * D_H * D_H
bytes_per_step = D_H * D_H * BF16_BYTES
i_rnn = flops_per_step / bytes_per_step

flops_total = S * flops_per_step
bytes_total_transformer = bytes_per_step  # same weight matrix, read ONCE for all S steps
i_transformer = flops_total / bytes_total_transformer

print("\n-- FLOP/byte accounting (bf16 storage, constants.md S6.5) --")
print(f"per-step:   FLOPs = 2*{D_H}^2 = {flops_per_step:,} = {flops_per_step/1e6:.2f} MFLOP"
      f"   bytes = {D_H}^2*{BF16_BYTES} = {bytes_per_step:,} = {bytes_per_step/1e6:.2f} MB")
print(f"I_RNN         = {flops_per_step/1e6:.2f} MFLOP / {bytes_per_step/1e6:.2f} MB = {i_rnn:.1f} FLOP/byte")
print(f"I_transformer = ({S}*{flops_per_step/1e6:.2f} MFLOP) / {bytes_per_step/1e6:.2f} MB = {i_transformer:.1f} FLOP/byte")

assert abs(flops_per_step - 33_554_432) < 1, f"FLOPs/step should be 33,554,432, got {flops_per_step}"
assert abs(bytes_per_step - 33_554_432) < 1, f"bytes/step should be 33,554,432, got {bytes_per_step}"
assert abs(i_rnn - 1.0) < 1e-9, f"I_RNN should be exactly 1.0 FLOP/byte, got {i_rnn}"
assert abs(i_transformer - 2048.0) < 1e-9, f"I_transformer should be exactly 2048 FLOP/byte, got {i_transformer}"
print("[OK] matches the page's frozen numbers exactly: 33.55 MFLOP, 33.55 MB, I=1.0 vs I=2048.")

# --------------------------------------------------------------------------
# Now TIME it -- the actual point of this script. Same weight matrix, same
# total FLOPs, two execution strategies: a real data dependency vs a batch.
# fp32 for CPU-portable, seconds-scale timing (see SAFETY note above); the
# FLOP/byte figures above stand regardless, they describe bf16 storage.
# --------------------------------------------------------------------------
W = torch.randn(D_H, D_H)
h0 = torch.randn(D_H)
X = torch.randn(S, D_H)  # S independent "input" vectors for the batched path

print(f"\n-- timing: {S} sequential matrix-vector products vs 1 batched matmul --")
print(f"   both do S x (2 x {D_H}^2) = {flops_total:,} FLOPs against the SAME {W.shape} weight matrix")

# THE RNN PATH: a genuine dependency chain. h is reassigned each iteration and
# fed straight back in -- step i literally cannot start before step i-1 finishes.
h = h0.clone()
t0 = time.perf_counter()
for _ in range(S):
    h = W @ h
t_loop = time.perf_counter() - t0
loop_result = h.clone()

# THE TRANSFORMER PATH: one kernel call, S independent matrix-vector products
# against the same W, issued as a single (S, D_H) @ (D_H, D_H) matmul.
t0 = time.perf_counter()
Y = X @ W.T
t_matmul = time.perf_counter() - t0

print(f"\nsequential loop  ({S} dependent matvecs): {t_loop:.4f} s")
print(f"batched matmul   (1 kernel, {S}x the work of one matvec): {t_matmul:.4f} s")
print(f"speedup: {t_loop / t_matmul:.1f}x faster, doing the IDENTICAL total FLOPs")

assert t_loop > t_matmul, (
    f"the whole point: the loop ({t_loop:.4f}s) should be wall-clock-slower than the "
    f"batched matmul ({t_matmul:.4f}s) despite doing the SAME total FLOPs -- if this "
    f"fails, something about this box's BLAS threading is unusual, but the physics "
    f"(a data dependency blocks pipelining/parallelism) does not change."
)
print(f"[OK] the loop is wall-clock-slower ({t_loop:.4f}s > {t_matmul:.4f}s) -- same FLOPs, "
      f"different dependency structure.")

print("\n" + "=" * 72)
print("Same weight matrix. Same total FLOPs (both S x 33.55 MFLOP). One path can only")
print("run one step at a time because each step NEEDS the last one's output; the other")
print("has no such dependency and is issued as a single kernel. The RNN didn't lose on")
print("arithmetic -- it lost because a matrix-vector product wastes a GPU and a")
print("matrix-matrix product doesn't, and dependency is what forces the former.")
print("=" * 72)
