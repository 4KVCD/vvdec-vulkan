# Changelog of vvdec-vulkan

## v1.2.1 (2026-11-15)

### Bug fixes

- Fixed the first pictures of a stream waiting minutes on Windows when the shader cache was empty (a first run, or
  after a driver update) and other programs kept the CPU busy: the shaders no longer compile at a lower priority than
  the decoding that waits for them.
- Fixed a rare crash in NVIDIA's driver while the shaders compiled for the first time (about one first run in five on
  an RTX 5090): on NVIDIA's GPUs the shaders now compile one at a time. A first run with an empty shader cache takes
  about 30 s longer there; later runs are as fast as before.
- Fixed a decoder's close taking over 10 s on Intel's GPUs with an empty shader cache: the shaders of the GPU's parse
  are no longer compiled ahead where the CPU parses.

## v1.2 (2026-11-10)

- The GPU's parse (`VVDEC_GPU_PARSE=2`) decodes 4K streams with tiles and without WPP faster, at high bitrates
  6-19% (Nocturne 4K, 300 pictures: RTX 5090 82 -> 97 fps, RTX 3070 44 -> 47 fps).
- The GPU's parse starts a 4K stream at a high bitrate 15-32% faster (Nocturne 4K at 245 Mb/s, 80 pictures: RTX 5090
  4.4 -> 3.8 s, RTX 3070 7.1 -> 5.4 s, Radeon 8060S 3.0 -> 2.6 s), with less GPU memory.

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
