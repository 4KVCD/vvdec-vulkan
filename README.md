# VVdeC on Vulkan: an H.266/VVC decoder that runs on the GPU

An experimental fork of [VVdeC](https://github.com/fraunhoferhhi/vvdec) 3.2.1, Fraunhofer HHI's VVC decoder, that
decodes on the GPU with Vulkan compute shaders. It is not an official Fraunhofer project.

- **Bit-exact.** Every output frame is identical to upstream VVdeC's. The check decodes, in both GPU modes, the 250
  JVET conformance streams that upstream VVdeC decodes correctly, 34 streams made to exercise particular coding
  tools, and long film encodes, and compares every frame: 609 of 609 decodes pass on the AMD GPU below. On Intel
  Graphics, the check's wide tier (every conformance stream and the films' first 240 to 600 frames, in both modes;
  the tool streams were not on that PC) passes 512 of 512 decodes. On an NVIDIA GeForce RTX 3070, four films (600
  to 1200 frames each, every frame compared) pass in both modes; the conformance streams were not run there.
- **How it compares with the CPU decoder depends on the CPU beside the GPU.** On a laptop's Radeon 780M, `--gpu 1`
  decodes 4K films 1.23 to 1.71 times as fast as its 8-core CPU and 1080p films 0.76 to 1.04 times. On the Intel
  Graphics of a Core Ultra 9 285K, `--gpu 1` decodes 4K films at about 111 fps and 1080p films at 329 to 434 fps,
  while the 24-core CPU decoder is 2.2 to 3.3 times faster but takes about twice the energy a frame (see Speed). On
  an NVIDIA GeForce RTX 3070 beside an 8-core Core i7-10700, `--gpu 1` decodes 4K films 1.83 to 1.85 times as fast as
  the CPU decoder and 1080p films 1.25 to 1.41 times, using about one CPU core instead of twelve.
- **The desktop stays smooth while the GPU decodes.** The decoder's GPU queues run at low priority, and on Intel's
  GPUs on the compute engine, away from the 3D engine that the desktop's compositor uses. Decoding 4K on Intel
  Graphics, the compositor's work waited past a 60 Hz frame 0.6% of the time, against 14.3% before.
- **Tested on four GPUs, on Windows 11:** the AMD Radeon 780M integrated in a Ryzen 7 8845HS (driver
  32.0.31041.1004), with the full check; an NVIDIA GeForce RTX 5090 (driver 616.92), with seven film cuts; the
  Intel Graphics of a Core Ultra 9 285K (driver 101.8991), with the wide tier; and an NVIDIA GeForce RTX 3070 (driver
  616.92), with four films. The Intel changes (since 1972535) have not been re-checked on AMD or the RTX 5090, and
  the NVIDIA changes (the GPU parse on NVIDIA, more pictures in parse on discrete GPUs) only on the RTX 3070.

## What runs on the GPU

`vvdecapp --gpu 1` (in the library, `vvdecParams::gpu = 1`) turns the GPU on. There are two GPU modes:

| | `--gpu 1` | `--gpu 1` with `VVDEC_GPU_PARSE=2` |
|---|---|---|
| Parsing the slice data (CABAC) | CPU | GPU |
| Motion derivation, deblocking parameters and edges, the order in which intra blocks are predicted | GPU (CPU on Intel's GPUs) | GPU |
| Residuals and transforms, intra and inter prediction, LMCS, deblocking, SAO, ALF and CC-ALF, film grain | GPU | GPU |

In both modes the CPU reads the parameter sets, the picture and slice headers and the reference picture lists, and
records and submits the GPU's work. The decoded frames are copied back to host memory and come out of the usual
vvdec API.

The GPU path covers every coding tool that upstream VVdeC supports, among them affine motion with PROF, SbTMVP, GPM,
MMVD, DMVR, BDOF, CIIP, BCW, weighted prediction, IBC, reference picture resampling, wraparound, MIP, ISP, MRL,
CCLM, LFNST, MTS, SBT, dependent quantization, joint chroma coding, transform skip and BDPCM, in 4:0:0, 4:2:0, 4:2:2
and 4:4:4 at 8 and 10 bits. Streams that use adaptive colour transform, which upstream VVdeC does not decode
correctly either, are decoded on the CPU.

## Speed

The fps leave start-up out: they count from the first output frame to the end of decoding. Each number is the median
of 3 runs, with the three configurations run in turn, with 16 decoder threads.

### AMD Radeon 780M (Ryzen 7 8845HS laptop, 8 cores)

| Video | Size | Substreams a picture | GPU parse (`VVDEC_GPU_PARSE=2`) | `--gpu 1` | CPU decoder (`--gpu 0`) | `--gpu 1` against the CPU |
|---|---|---|---|---|---|---|
| 4K film, VVenC slower preset | 3840x1608 | 26 | 202.1 fps | 202.1 fps | 141.7 fps | 1.43x |
| 4K film | 3840x2160 | 34 | 123.9 fps | 156.6 fps | 110.4 fps | 1.42x |
| 4K film | 3840x1608 | 1 | 182.8 fps | 223.2 fps | 180.8 fps | 1.23x |
| 4K film with film grain synthesis | 3840x2160 | 2 | 166.3 fps | 170.0 fps | 99.6 fps | 1.71x |
| 1080p film, VVenC slower preset | 1920x1080 | 18 | 457.4 fps | 505.1 fps | 487.2 fps | 1.04x |
| 1080p film | 1920x1080 | 18 | 468.3 fps | 522.5 fps | 502.2 fps | 1.04x |
| 1080p film | 1920x1080 | 2 | 343.1 fps | 526.0 fps | 525.0 fps | 1.00x |
| 1080p film | 1920x1080 | 1 | 296.4 fps | 363.6 fps | 480.8 fps | 0.76x |

All eight films are 10-bit. On each of them `--gpu 1` is at least as fast as the GPU parse.

### NVIDIA GeForce RTX 3070 (Core i7-10700, 8 cores, PCIe 3.0)

| Video | Size | Substreams a picture | GPU parse (`VVDEC_GPU_PARSE=2`) | `--gpu 1` | CPU decoder (`--gpu 0`) | `--gpu 1` against the CPU |
|---|---|---|---|---|---|---|
| 4K film, VVenC slower preset | 3840x1608 | 26 | 187.8 fps | 261.9 fps | 143.1 fps | 1.83x |
| 4K film | 3840x2160 | 1 | 170.4 fps | 243.1 fps | 131.5 fps | 1.85x |
| 1080p film, VVenC slower preset | 1920x1080 | 18 | 482.1 fps | 645.5 fps | 457.3 fps | 1.41x |
| 1080p film | 1920x1080 | 1 | 459.5 fps | 698.4 fps | 559.0 fps | 1.25x |

The decoder used 0.4 to 0.7 CPU cores with the GPU parse, 0.7 to 1.5 with `--gpu 1` and 12 to 14 as the CPU
decoder. With the GPU parse the GPU drew about 120 W on the 4K films (nvidia-smi), well below this card's 270 W limit.

- **The GPU parse is 25 to 34% slower than `--gpu 1` here.** It runs on NVIDIA since 7275718 (before, the decoder
  turned it off there and parsed on the CPU). Discrete GPUs parse 32 to 64 pictures at a time, integrated ones 8 to
  16.
- **A larger GPU does not make it much faster.** The speed is set by chains of work that wait on one another rather
  than by the GPU's size: the motion derivation runs as a wavefront over the CTUs (3840x2160 with 64x64 CTUs: about
  126 steps one after another), and each picture's motion derivation waits for its colocated picture's first pass (TMVP).
- **The bus caps 4K.** Every decoded frame is copied to host memory over PCIe: a 3840x2160 10-bit frame (24.9 MB)
  took 2.0 to 2.2 ms on this PC, which caps such films at about 465 to 500 fps however fast the rest is.

### Intel Graphics (Core Ultra 9 285K, 24 cores)

| Video | Size | Substreams a picture | GPU parse (`VVDEC_GPU_PARSE=2`) | `--gpu 1` | CPU decoder (`--gpu 0`) | `--gpu 1` against the CPU |
|---|---|---|---|---|---|---|
| 4K film, VVenC slower preset | 3840x1608 | 26 | 72.6 fps | 111.9 fps | 300.2 fps | 0.37x |
| 4K film | 3840x2160 | 1 | 58.2 fps | 111.3 fps | 243.4 fps | 0.46x |
| 1080p film, VVenC slower preset | 1920x1080 | 18 | 170.8 fps | 328.7 fps | 1,083.2 fps | 0.30x |
| 1080p film | 1920x1080 | 1 | 129.4 fps | 433.9 fps | 1,220.7 fps | 0.36x |

The processor package (CPU and GPU) drew 19 to 26 W decoding with `--gpu 1` and 114 to 124 W with the CPU decoder:
the GPU path took 46 to 52% of the CPU decoder's energy a frame (4K: 176 and 237 mJ against 384 and 471 mJ), and left
the CPU mostly free. Before the Intel changes, `--gpu 1` decoded the 4K film 3840x2160 at 35 fps and the 1080p one
at 95 fps on this GPU.

On Intel's GPUs the CPU derives the motion and the deblocking parameters (`VVDEC_GPU_FRONT=1`: the GPU), and the
shaders that allow it run in waves of 16 threads. The GPU parse needs the GPU front end, so `VVDEC_GPU_PARSE=2` keeps
it on the GPU.

Where it is slow:

| Conformance stream | Size | GPU parse | `--gpu 1` | CPU decoder |
|---|---|---|---|---|
| LFNST_D_3 (all intra) | 1920x1080 | 54.7 fps | 98.0 fps | 172.2 fps |
| TEMPSCAL_A_4 | 832x480 | 250.4 fps | 464.2 fps | 1,317.9 fps |
| TREE_A_3 (all intra) | 768x384 | 183.6 fps | 475.9 fps | 1,199.5 fps |
| BOUNDARY_A_3 (the picture size changes) | up to 376x376 | 534.8 fps | 1,294.1 fps | 4,406.5 fps |

- **Small pictures.** Each picture goes through a chain of GPU passes that run one after another. On a small
  picture each pass has little work, so the time goes into the passes waiting on one another while the GPU stays
  mostly idle.
- **Intra pictures.** Intra blocks are predicted in waves, each depending on the one before, and the GPU works out
  those waves one block after another along each row of CTUs. Pictures that are mostly intra therefore run largely
  in sequence.
- **The GPU parse.** Arithmetic decoding is serial within a substream (a slice, a tile, or a row of CTUs with
  WPP), and each substream is parsed by one GPU workgroup, so a picture's parse can spread over only as many
  workgroups as it has substreams. In the tests above the GPU parse is never faster than `--gpu 1`: it is up to 35%
  slower on the films and 44 to 61% slower on the four conformance streams.
- **Start-up** (left out of the fps above) takes longer than the CPU decoder's: creating the Vulkan device, loading
  the compiled pipelines and filling the GPU's pipeline of pictures. The first run of a new build also compiles the
  shaders, about 7 s on this laptop.

## Building (Windows)

You need Visual Studio 2022 with the C++ tools, CMake, Ninja, the [Slang](https://github.com/shader-slang/slang)
compiler 2026.10.2 (`slangc.exe`) and the Khronos [Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers)
(version 1.4.364 was used). Then run:

```bat
scripts\build_vulkan.bat path\to\slangc.exe path\to\Vulkan-Headers\include
```

This builds Release with Ninja into `build\vulkan`; the programs (`vvdecapp.exe`, `vvdecbatch.exe`) land in
`bin\release-static`. The script runs CMake with `-DVVDEC_ENABLE_VULKAN=ON -DVVDEC_SLANGC=<slangc>
-DVVDEC_VULKAN_HEADERS=<include dir>`; the shaders are compiled to SPIR-V and embedded in the program. The same CMake
options also build with GCC 16.2 (MinGW-w64), adding `-DCMAKE_EXE_LINKER_FLAGS=-static` for programs that need no
MinGW DLLs. Other platforms have not been built.

At run time the decoder loads the Vulkan loader (`vulkan-1.dll`). It needs a Vulkan 1.2 device with buffer device
addresses, 8-, 16- and 64-bit integers in shaders, and 8- and 16-bit storage buffer access. The GPU parse and the
front end's own queue also need timeline semaphores and an asynchronous compute queue; without them the CPU parses
and the front end runs on the main queue. Keeping the compiled pipelines between runs needs Vulkan 1.3 with
`VK_KHR_pipeline_binary`.

## Running

```sh
vvdecapp -b input.266 -o output.yuv --gpu 1
```

| Environment variable | Effect |
|---|---|
| `VVDEC_GPU_PARSE=2` | the GPU parses the slice data too |
| `VVDEC_GPU_DEVICE=<n>` | use Vulkan device number `n` (default: the first discrete GPU, else the first integrated one, else the first device) |
| `VVDEC_GPU_CACHE=<file>` | where the compiled pipelines are kept (default: `%LOCALAPPDATA%\vvdec` on Windows); `0` keeps none |
| `VVDEC_GPU_PRECOMPILE=0` | compile each shader when it is first used instead of in the background at start |
| `VVDEC_GPU_FRONT=0` | the CPU derives the motion and the deblocking parameters and builds the job lists (the default on Intel's GPUs; `VVDEC_GPU_FRONT=1` keeps them on the GPU) |
| `VVDEC_GPU_ENGINE=compute` | the GPU's work on its compute engine (the default on Intel's GPUs); `3d`: on the graphics engine (the default elsewhere) |
| `VVDEC_GPU_PRIORITY=normal` | the GPU queues at normal priority (default: low, so the desktop's GPU work goes first) |
| `VVDEC_GPU_PROFILE=1` | at exit, print the GPU time of each pass, each shader's registers (AMD) or statistics (other GPUs) and the GPU memory peaks |
| `VVDEC_GPU_WAVE=<n>`, `VVDEC_GPU_WAVE_INTER=<n>` | tuning: the wave size of the shaders that work with any (`0`: the driver's choice) |
| `VVDEC_GPU_MIDER=1` | debugging: compare the GPU's motion, deblocking parameters and job lists with the CPU's |
| `VVDEC_GPU_PARSE=1` | debugging: the CPU parses, and the GPU's parse of each picture is compared with it |

## Documentation and tools

- [`source/Lib/VulkanLib/README.md`](source/Lib/VulkanLib/README.md): the design (the paths, the queues, a
  picture's journey through the tasks and passes). Each source file and shader begins with a description of its part.
- [`tools/gpu-dev/`](tools/gpu-dev/): the checks and benchmarks used during development (Python 3). `fastcheck.py`
  decodes the conformance and tool streams and compares every frame; `ab_steady.py` measures steady-state speed;
  `speedsweep.py` times every video against the CPU decoder; `uiprobe` measures how long the desktop's GPU work
  waits while a decode runs. The test streams are not in the repository.

## License

The same as upstream VVdeC: see [LICENSE.txt](./LICENSE.txt). VVdeC® is a registered trademark of the
Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V. This fork is not affiliated with or endorsed by
Fraunhofer.

---

*The upstream VVdeC README follows. Its list of supported platforms is upstream's, for the CPU decoder; the GPU
decoder has only been built on Windows x64.*

# Upstream VVdeC

![VVdeC Logo](https://github.com/fraunhoferhhi/vvdec/wiki/img/VVdeC_RGB_small.png)

VVdeC, the Fraunhofer Versatile Video Decoder, is a fast software H.266/VVC decoder implementation supporting **all features** of the **VVC Main10** profile.

## Supported architectures


<table>
  <tr>
    <th colspan="2" align="center" valign="center"><strong>Windows</strong></th>
    <th colspan="2" align="center" valign="center"><strong>Linux</strong></th>
    <th colspan="2" align="center" valign="center"><strong>MacOS X</strong></th>
    <th colspan="2" align="center" valign="center"><strong>Android</strong></th>
    <th colspan="2" align="center" valign="center"><strong>iOS</strong></th>
    <th colspan="2" align="center" valign="center"><strong>Browser (WASM)</strong></th>
  </tr>
  <tr>
    <td>Win32</td>
    <td>:white_check_mark:</td>
    <td>x86</td>
    <td>:white_check_mark:</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td>Edge</td>
    <td>:white_check_mark:</td>
  </tr>
  <tr>
    <td>x64</td>
    <td>:white_check_mark:</td>
    <td>x86_64</td>
    <td>:white_check_mark:</td>
    <td>x64</td>
    <td>:black_square_button:</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td>Firefox</td>
    <td>:white_check_mark:</td>
  </tr>
  <tr>
    <td>armv7</td>
    <td>:x:</td>
    <td>armv7</td>
    <td>:white_check_mark:</td>
    <td></td>
    <td></td>
    <td>armv7</td>
    <td>:white_check_mark:</td>
    <td>armv7</td>
    <td>:black_square_button:</td>
    <td>Chrome</td>
    <td>:white_check_mark:</td>
  </tr>
  <tr>
    <td>aarch64</td>
    <td>:white_check_mark:</td>
    <td>aarch64</td>
    <td>:white_check_mark:</td>
    <td>arm64</td>
    <td>:white_check_mark:</td>
    <td>aarch64</td>
    <td>:white_check_mark:</td>
    <td>arm64</td>
    <td>:black_square_button:</td>
    <td>Safari</td>
    <td>:x:</td>
  </tr>
</table>

:white_check_mark: tested and works :black_square_button: needs testing (might already work) :x: does not work

Other architectures and platforms might work, see the [Wiki](https://github.com/fraunhoferhhi/vvdec/wiki#supported-architectures).

## Information

See the [Wiki-Page](https://github.com/fraunhoferhhi/vvdec/wiki) for more information:

* [Build information](https://github.com/fraunhoferhhi/vvdec/wiki/Build)
* [Usage documentation](https://github.com/fraunhoferhhi/vvdec/wiki/How-to-use-VVdeC)
* [License](https://github.com/fraunhoferhhi/vvdec/wiki/License)
* [Publications](https://github.com/fraunhoferhhi/vvdec/wiki/Publications)
* [Version history](https://github.com/fraunhoferhhi/vvdec/wiki/Changelog)

## Build

VVdeC uses CMake to describe and manage the build process. A working [CMake](https://cmake.org/) installation is required to build the software. In the following, the basic build steps are described. Please refer to the [Wiki](https://github.com/fraunhoferhhi/vvdec/wiki/Build) for the description of all build options.

### How to build using CMake?

To build using CMake, create a `build` directory and generate the project:

```sh
mkdir build
cd build
cmake .. <build options>
```

To actually build the project, run the following after completing project generation:

```sh
cmake --build .
```

For multi-configuration projects (e.g. Visual Studio or Xcode) specify `--config Release` to build the release configuration.

### How to build using GNU Make?

On top of the CMake build system, convenience Makefile is provided to simplify the build process. To build using GNU Make please run the following:

```sh
make install-release <options>
```

Use the option `enable-bitstream-download=1` in the make command to download the VVC conformance bitstreams for testing.

Other supported build targets include `configure`, `release`, `debug`, `relwithdebinfo`, `test`,  and `clean`. Refer to the Wiki for a full list of supported features.

## Citing

Please use the following citation when referencing VVdeC in literature:

```bibtex
@InProceedings{VVdeC,
  author    = {Wieckowski, Adam and Hege, Gabriel and Bartnik, Christian and Lehmann, Christian and Stoffers, Christian and Bross, Benjamin and Marpe, Detlev},
  booktitle = {Proc. IEEE International Conference on Image Processing (ICIP)},
  date      = {2020},
  title     = {Towards A Live Software Decoder Implementation For The Upcoming Versatile Video Coding (VVC) Codec},
  doi       = {10.1109/ICIP40778.2020.9191199},
  pages     = {3124-3128},
}

```

## Contributing

Feel free to contribute. To do so:

* Fork the current-most state of the master branch
* Apply the desired changes
* For non-trivial contributions, add your name to [AUTHORS.md](./AUTHORS.md)
* Create a pull-request to the upstream repository

## License

Please see [LICENSE.txt](./LICENSE.txt) file for the terms of use of the contents of this repository.

For more information, please contact: vvc@hhi.fraunhofer.de

**Copyright (c) 2018-2026 Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V. & The VVdeC Authors.**

**All rights reserved.**

**VVdeC® is a registered trademark of the Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V.**
