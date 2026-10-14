# VVdeC on Vulkan: an H.266/VVC decoder that runs on the GPU

An experimental fork of [VVdeC](https://github.com/fraunhoferhhi/vvdec) 3.2.1, Fraunhofer HHI's VVC decoder, that
decodes on the GPU with Vulkan compute shaders. It is not an official Fraunhofer project.

- **Bit-exact.** Every output frame is identical to upstream VVdeC's. The check decodes, in both GPU modes, the 250
  JVET conformance streams that upstream VVdeC decodes correctly, 34 streams made to exercise particular coding
  tools, and long film encodes, and compares every frame: 609 of 609 decodes pass.
- **Faster than the CPU decoder at 4K, slower at 1080p and below**, on the one laptop it was tested on: 1.10 to
  1.85 times the CPU decoder's speed on 4K films, 0.81 to 0.96 times on 1080p films (see Speed).
- **Tested on one GPU only:** the AMD Radeon 780M integrated in a Ryzen 7 8845HS, on Windows 11 with driver
  32.0.31041.1004. NVIDIA and Intel GPUs have not been tried.

## What runs on the GPU

`vvdecapp --gpu 1` (in the library, `vvdecParams::gpu = 1`) turns the GPU on. There are two GPU modes:

| | `--gpu 1` | `--gpu 1` with `VVDEC_GPU_PARSE=2` |
|---|---|---|
| Parsing the slice data (CABAC) | CPU | GPU |
| Motion derivation, deblocking parameters and edges, the order in which intra blocks are predicted | GPU | GPU |
| Residuals and transforms, intra and inter prediction, LMCS, deblocking, SAO, ALF and CC-ALF, film grain | GPU | GPU |

In both modes the CPU reads the parameter sets, the picture and slice headers and the reference picture lists, and
records and submits the GPU's work. The decoded frames are copied back to host memory and come out of the usual
vvdec API.

Every coding tool that upstream VVdeC supports runs on the GPU, among them affine motion with PROF, SbTMVP, GPM,
MMVD, DMVR, BDOF, CIIP, BCW, weighted prediction, IBC, reference picture resampling, wraparound, MIP, ISP, MRL,
CCLM, LFNST, MTS, SBT, dependent quantization, joint chroma coding, transform skip and BDPCM, in 4:0:0, 4:2:0, 4:2:2
and 4:4:4 at 8 and 10 bits. The one exception is adaptive colour transform, which upstream VVdeC does not decode
correctly either: streams that use it are decoded on the CPU.

## Speed

Measured on the laptop above with 16 decoder threads: one decode at a time, the whole decode including start-up,
one run each. Power was not controlled: this laptop runs anywhere between about 15 and 38 W, which by itself moves
the fps by up to about 25%, so compare the columns of a row rather than the rows.

| Video | Size | Frames | GPU parse (`VVDEC_GPU_PARSE=2`) | CPU parse (`--gpu 1`) | CPU decoder (`--gpu 0`) |
|---|---|---|---|---|---|
| Film, VVenC slower preset, WPP | 3840x1608 | 300 | 113.6 fps | 135.0 fps | 102.5 fps |
| Film with film grain synthesis | 3840x2160 | 300 | 88.6 fps | 125.0 fps | 67.7 fps |
| Film, VVenC medium preset | 3840x2160 | 300 | 75.6 fps | 105.7 fps | 81.1 fps |
| Film, VVenC slower preset, no WPP | 3840x1608 | 300 | 82.9 fps | 141.3 fps | 127.9 fps |
| Film, VVenC slower preset | 1920x1080 | 600 | 242.3 fps | 295.5 fps | 317.0 fps |
| Film, VVenC medium preset | 1920x1080 | 600 | 258.4 fps | 312.4 fps | 324.2 fps |
| Film, one substream a picture | 1920x1080 | 600 | 209.6 fps | 319.6 fps | 396.3 fps |

On every film, `--gpu 1` (the CPU parses) is the faster of the two GPU modes.

Where it is slow:

- **Small pictures.** The GPU spends about 0.8 to 2.5 ms on each picture in passes that wait on one another, so at
  832x480 and below `--gpu 1` decodes about 400 to 1,250 fps where the CPU decoder does 900 to 4,800.
- **All-intra 1080p.** Intra blocks are predicted in waves, each of which depends on the one before, so much of the
  work runs in sequence: 77 fps against the CPU decoder's 123 on the LFNST_D_3 conformance stream.
- **The GPU parse when a picture has few substreams.** Arithmetic decoding is serial within a substream, and one GPU
  wave does it roughly 25 times slower than a CPU core. WPP or many tiles give a picture enough substreams to keep
  the GPU busy. With one substream a picture, or at very high bitrates, `VVDEC_GPU_PARSE=2` decodes 2 to 17 times
  slower than `--gpu 1`.
- **Start-up.** With the Vulkan device already running, the first frame came after a median of 0.1 s with
  `--gpu 1` and 0.5 s with the GPU parse (the CPU decoder: 0.04 s). The first run of a new build also compiles the
  shaders, about 7 s; the compiled pipelines are kept on disk for later runs.

## Building (Windows)

You need Visual Studio 2022 with the C++ tools, CMake, Ninja, Python 3, the
[Slang](https://github.com/shader-slang/slang) compiler 2026.10.2 (`slangc.exe`) and the Khronos
[Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers), 1.4.3xx or newer. Then run:

```bat
scripts\build_vulkan.bat path\to\slangc.exe path\to\Vulkan-Headers\include
```

This builds Release with Ninja into `build\vulkan` and puts `vvdecapp.exe` in `bin\release-static`. The script runs
CMake with `-DVVDEC_ENABLE_VULKAN=ON -DVVDEC_SLANGC=<slangc> -DVVDEC_VULKAN_HEADERS=<include dir>`; the shaders are
compiled to SPIR-V and embedded in the program. Other platforms have not been built.

At run time the decoder loads the Vulkan loader (`vulkan-1.dll`). It needs a Vulkan 1.2 device with buffer device
addresses and 8-, 16- and 64-bit integers in shaders.

## Running

```sh
vvdecapp -b input.266 -o output.yuv --gpu 1
```

| Environment variable | Effect |
|---|---|
| `VVDEC_GPU_PARSE=2` | the GPU parses the slice data too |
| `VVDEC_GPU_DEVICE=<n>` | use Vulkan device number `n` (default: the first discrete GPU, else the first integrated one) |
| `VVDEC_GPU_CACHE=<file>` | where the compiled pipelines are kept (default: under `%LOCALAPPDATA%\vvdec`); `0` keeps none |
| `VVDEC_GPU_PRECOMPILE=0` | compile each shader when it is first used instead of in the background at start |
| `VVDEC_GPU_FRONT=0` | the CPU derives the motion and the deblocking parameters and builds the job lists |
| `VVDEC_GPU_PROFILE=1` | at exit, print the GPU time of each pass, each shader's registers and the GPU memory peaks |
| `VVDEC_GPU_MIDER=1` | debugging: compare the GPU's motion, deblocking parameters and job lists with the CPU's |
| `VVDEC_GPU_PARSE=1` | debugging: the CPU parses, and the GPU's parse of each picture is compared with it |

## Documentation and tools

- [`source/Lib/VulkanLib/README.md`](source/Lib/VulkanLib/README.md): the design (the paths, the queues, a
  picture's journey through the tasks and passes). Each source file and shader begins with a description of its part.
- [`VULKAN_HANDOFF.md`](VULKAN_HANDOFF.md): development notes written for continuing the work; parts are out of
  date.
- [`tools/gpu-dev/`](tools/gpu-dev/): the checks and benchmarks used during development. `fastcheck.py` decodes the
  conformance and tool streams and compares every frame; `speedsweep.py` times every video against the CPU decoder.
  The test streams are not in the repository.

## License

The same as upstream VVdeC: see [LICENSE.txt](./LICENSE.txt). VVdeC® is a registered trademark of the
Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V. This fork is not affiliated with or endorsed by
Fraunhofer.

---

*The upstream VVdeC README follows.*

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
