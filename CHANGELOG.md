# Changelog of vvdec-vulkan

## v1.2.5 (2026-11-26)

### Changes

- Film grain on ARM CPUs runs on Neon code (upstream VVdeC PR #450).

### Bug fixes

- Fixed film grain on 4:4:4, 4:2:2 and 4:0:0 video, on the CPU and the GPU: 4:4:4 chroma got far too strong grain,
  4:2:2 left about 41 % of the chroma samples without grain, and 4:0:0 video stopped decoding at the first picture
  with grain (upstream VVdeC PR #450).
- Fixed wrong film grain where a scaling factor is 128 or more, in the x86 SIMD code (upstream VVdeC PR #449).

## v1.2.4 (2026-11-26)

### Bug fixes

- Fixed film grain changing with where decoding started: each picture's grain now comes from its own coded data, so
  a picture gets the same grain after a seek, a cut or a restart as in a decode from the start (upstream VVdeC PR
  #486).
- Fixed a colour component keeping the grain of an earlier film grain message when a new message left it out.

## v1.2.3 (2026-11-18)

### Bug fixes

- Fixed closing the decoder taking up to 11 s on Intel GPUs whose driver had not cached the shaders yet: the close
  no longer waits for shaders compiled ahead of time that the stream did not use.

## v1.2.2 (2026-11-17)

### Bug fixes

- Fixed the shaders compiling again from the start (about 45 s on an RTX 5090) after a first run that was ended while
  they compiled: the compiled shaders are now saved as they are made, not only when the decoder closes.
- Fixed the shaders compiling again at every run where the user's cache folder (AppData\Local, or ~/.cache) was
  missing: the cache folder is now made with every missing folder above it.

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
