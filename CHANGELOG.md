# Changelog of vvdec-vulkan

## v1.1 (2026-11-02)

- Hand-over: decoded pictures copied on the GPU straight into another Vulkan device's buffers on the same GPU, or
  into imported host memory, with no download (`vvdec_handover_*`); 10-bit and 8-bit, with the CPU's parse or the
  GPU's parse ahead.
- Film grain synthesis with the hand-over.
- RTX 5090 with the CPU's parse (`--gpu 1`) 2.3-2.4x faster at 4K: Beekeeper QP 16 55 -> 127 fps, with film grain
  65 -> 158 fps.

### Bug fixes

- A damaged slice no longer loses the GPU device with the GPU's parse.
- A damaged picture is output concealed again with the GPU's parse, instead of missing.
- Film grain on integrated GPUs (Intel, Radeon) matches the CPU's output again.
