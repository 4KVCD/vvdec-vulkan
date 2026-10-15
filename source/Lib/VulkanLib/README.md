# VulkanLib: reconstructing VVC pictures on the GPU

With `vvdecParams::gpu` set (`vvdecapp --gpu 1`), vvdec reconstructs pictures with Vulkan compute shaders. The
output is bit-exact with the CPU decoder. The CPU handles the parameter sets, picture and slice headers and reference
picture lists, and records and submits the GPU work. It also parses the slice data (CABAC), unless
`VVDEC_GPU_PARSE=2` hands that to the GPU too (see "The GPU parse" below). This document is the overview; each
source file and each shader starts with a description of its own part.

## The three paths

The path is chosen per picture in `DecLibRecon::decompressPicture`.

1. **GPU front end** (the default). The CPU's CTU tasks only parse. They write a compact record of every CU and
   TU (`GpuFrontEnd.h`: `CuRecord`, `TuRecord`, `CtuRecord`). From those records the GPU derives the motion
   (`mider.slang`), the deblocking parameters (`dbparams.slang`), the job lists of every reconstruction pass
   (`jobs.slang`), the order of the intra blocks (`intrasort.slang`) and the deblocking edges (`edges.slang`).
   It then reconstructs the picture.
2. **CPU front end** (`VVDEC_GPU_FRONT=0`). The CPU's CTU tasks derive the motion and the deblocking parameters as
   vvdec does, and build the job lists themselves (`DecCu::xGpu*Jobs`, `GpuRecon::listEdges`). The GPU runs only
   the reconstruction passes.
3. **CPU only**. Used for a picture with a feature the GPU path does not cover, and with `--gpu 0`.

`VVDEC_GPU_MIDER=1` runs path 2 and the GPU front end side by side, and compares the GPU's motion, deblocking
parameters, job lists and edges with the CPU's (`GpuRecon::compareFrontEnd`). It is a debugging aid.

## The GPU parse

`parse.slang` parses the slice data on the GPU: one workgroup of 32 threads per substream (a slice, a tile, or a
CTU row with WPP), in dispatches of a bounded number of bins each, resumed where they stopped (a single long
dispatch or submission would be cut short by the driver or reset by Windows). It writes what the CPU's parse
writes: the CU and TU records, the coefficient levels in the picture's host planes, and each CTU's SAO and ALF
parameters. The records are **staged** wherever the picture's counters give them room, each with its CTU and its
place in the CTU; `parsecompact.slang` then works out each CTU's first CU and TU in raster order (pass 0, once the
picture is parsed) and copies the records there (the **scatter**). The parse also adds up the room the front end's
job lists need (`gpu::frontCapacities`).

- `VVDEC_GPU_PARSE=1`: the check. The CPU parses as usual, and each picture is also parsed on the GPU, in its own
  submissions on the compute queue (`GpuRecon::recordParse`); the records, levels, SAO and ALF parameters and the
  job lists' room are compared with the CPU's (`GpuRecon::compareParse`).
- `VVDEC_GPU_PARSE=2`: the GPU parses instead of the CPU (with path 1). The CPU's parse task only keeps the slice
  data and marks the slice's CTUs parsed (`DecSlice::claimSlice`); once a picture's last slice is in, it goes to
  the **ParseService** (`GpuParseService.h`). Its thread parses many pictures ahead of their reconstruction at a
  time: each submission on the **parse queue** (an async compute queue of its own) holds a few dispatches of every
  picture it is parsing, side by side, and each picture's compaction pass 0. A picture whose parse is done unlocks
  `Picture::gpuParsed`; its front end then scatters the records from the picture's **parse slot** into its own
  buffers, waiting for the parse queue's timeline semaphore. The slot is reused once the picture's first part is
  done on the GPU.

## A picture's journey (path 1)

**Tasks** (`DecLibRecon`, on the decoder's thread pool):

| task | runs when | does |
|---|---|---|
| CTU tasks | the picture is parsed | write the CU and TU records (nothing with `VVDEC_GPU_PARSE=2`) |
| `filterParamsTask` | the picture is parsed (with `VVDEC_GPU_PARSE=2`: `gpuParsed`) | works out the SAO, ALF and deblocking parameters (`GpuRecon::prepareFilters`; with `VVDEC_GPU_PARSE=2` first the CTUs' SAO and ALF parameters from the GPU's parse) |
| `queueGpuTask` | its CTU tasks are done and every reference picture's GPU work is queued (`gpuRefReady`) | records and submits the picture's GPU work (`GpuRecon::startPicture`), then unlocks the picture's `gpuRefReady` |
| `computeDoneTask` | the picture's GPU work is queued | waits for the GPU's compute work (`GpuRecon::endCompute`), then unlocks `gpuComputeDone`: the decoder instance may take its next picture |
| `finishGpuTask` | `gpuComputeDone` | waits for the picture's download to host memory (`GpuRecon::endDownload`) |

**Queues and submissions** (`GpuRecon::startPicture`):

| queue | submission | contents | waits for |
|---|---|---|---|
| parse (`VVDEC_GPU_PARSE=2`) | the ParseService's | a few parse dispatches of each picture being parsed, their compaction pass 0 | the submission before (the service waits for each) |
| async compute (one per decoder instance) | front end, part 1 | the scatter of the GPU's parse's records (`VVDEC_GPU_PARSE=2`), motion, colocated motion, job lists | the first parts of the picture's colocated pictures (the timeline semaphore); the picture's parse (the parse queue's timeline semaphore) |
| async compute | front end, part 2 | intra order, deblocking parameters, deblocking edges | front end part 1 (same queue) |
| compute | first part | uploads of reference pictures the CPU made, residuals, inter prediction (DMVR writes its refined motion into the colocated motion) | front end part 1; then signals the timeline semaphore |
| compute | rest | intra prediction, LMCS's inverse mapping, deblocking, SAO, ALF and CC-ALF | front end part 2 |
| copy | download | the finished picture to host memory | the rest |

The next picture's front end reads the refined motion through TMVP, which is why it waits for its colocated
pictures' first parts.

**Pass order inside the picture**: residuals (`trafo.slang`) -> inter prediction (`inter.slang`) -> intra prediction
in dependency waves (`intra.slang`) -> LMCS inverse mapping (`lmcs.slang`) -> deblocking (`deblock.slang`) -> SAO,
in place on the CTUs that use it (`sao.slang`) -> ALF and CC-ALF (`alf.slang`). Film grain (`filmgrain.slang`, `GpuFilmGrain`) is applied to the
output frames, not to the reference pictures.

## Glossary

- **host plane / twin**: every picture plane lives in host memory that the GPU can reach (`hostPictureAllocator`),
  where vvdec's CPU code reads and writes it. Its **twin** is a copy in the GPU's own memory, where the GPU
  reconstructs the picture and keeps it as a reference picture.
- **scratch planes**: picture-sized GPU memory taken from a pool for ALF's output; after ALF they become the
  picture's twin.
- **first part / rest**: the two command buffers of a picture's work on the compute queue (see above).
- **front end**: everything between parsing and reconstruction: motion derivation, deblocking parameters, job
  lists, intra order, deblocking edges.
- **records**: the per-CU, per-TU and per-CTU words the CPU's parse writes for the GPU front end (`GpuFrontEnd.h`).
- **staged records, scatter**: the GPU parse's records where they were written, each with its CTU and its place in
  the CTU, and their copy to raster order (`parsecompact.slang`).
- **parse slot**: everything of one picture's parse on the GPU (`ParseSlot`): its inputs, its substreams' state, its
  staged records, its CTUs' counts and SAO and ALF parameters.
- **jobs**: the work items of a reconstruction pass (`GpuJobs.h`): `TrafoJob`, `InterJob`, `IntraJob`.
- **waves**: groups of intra jobs that may run at the same time; a job's wave is one more than the highest wave
  of the blocks it reads from.
- **intra bands**: intra blocks split into strips of 256 samples, the unit of work of `intra.slang`.
- **early rows**: with the CPU front end, the top half of a picture's CTU rows, whose residuals and inter blocks
  are submitted before the CPU finishes the rest (`GpuRecon::startEarlyRows`).
- **deferred predictions**: inter predictions kept aside for an intra job to finish (LMCS chroma scaling, CIIP).

## Environment variables

| variable | effect |
|---|---|
| `VVDEC_GPU_DEVICE=<n>` | use the Vulkan device with index n (default: the first discrete GPU, else the first integrated one) |
| `VVDEC_GPU_PROFILE=1` | print the GPU time of each pass and each shader's registers and memory at exit (without `VK_AMD_shader_info`: its statistics from `VK_KHR_pipeline_executable_properties`) |
| `VVDEC_GPU_ISA=<dir>` | with `VVDEC_GPU_PROFILE=1` on AMD: write each shader's machine code to `<dir>/<name>.isa` |
| `VVDEC_GPU_FRONT=0` | path 2: the CPU builds the GPU's jobs (the default on Intel's GPUs; `VVDEC_GPU_FRONT=1`: path 1) |
| `VVDEC_GPU_ENGINE=compute` | the compute queue in a compute-only family, the GPU's compute engine (the default on Intel's GPUs); `3d`: in the graphics family (the default elsewhere) |
| `VVDEC_GPU_PRIORITY=normal` | the queues at normal global priority (default: low, `VK_KHR_global_priority` where the device has it) |
| `VVDEC_GPU_WAVE=<n>`, `VVDEC_GPU_WAVE_INTER=<n>` | the wave size of the shaders that work with any, and of inter's variants (`0`: the driver's choice; default: 16 and the driver's on Intel's GPUs, 32 and the driver's elsewhere) |
| `VVDEC_GPU_MIDER=1` | path 2 plus the GPU front end, compared with the CPU's (debugging) |
| `VVDEC_GPU_ASYNC=0` | run the front end on the compute queue instead of an async compute queue |
| `VVDEC_GPU_PARSE=1` | the GPU parses the slice data too, compared with the CPU's parse (debugging) |
| `VVDEC_GPU_PARSE=2` | the GPU parses the slice data instead of the CPU (needs the async compute queues) |
| `VVDEC_GPU_CACHE=<file>` | where the compiled shaders are kept between runs (default: `%LOCALAPPDATA%\vvdec\` on Windows, `$XDG_CACHE_HOME/vvdec/` or `~/.cache/vvdec/` elsewhere; one file per GPU, made anew when the driver or the shaders change); `0`: none, every program's first run compiles them again (~12 s) |
| `VVDEC_FASTHASH=1` | (vvdecapp) `-md5` hashes the output with CRC-32C instead of MD5, for fast checks |

## Files

- `GpuContext.*`: the Vulkan device, its queues, memory allocation and the shaders' pipelines.
- `GpuPictureMemory.*`: the pictures' planes in host memory the GPU can reach, and their twins on the GPU.
- `GpuRecon.h`: `GpuRecon`, which records and submits a picture's GPU work, from the uploads to the download. Its
  functions are defined by area:
  - `GpuRecon.cpp`: construction, commands and submissions, the picture's planes and tables, and the order of a
    picture's work;
  - `GpuReconFrontEnd.cpp`: the GPU front end;
  - `GpuReconPrediction.cpp`: the residuals, inter prediction and intra prediction;
  - `GpuReconFilters.cpp`: LMCS's inverse mapping, deblocking, SAO and ALF;
  - `GpuReconParse.cpp`: whether the GPU parses a picture, and the check of its parse (`VVDEC_GPU_PARSE=1`);
  - `GpuFrontEndCheck.*`: the comparison of the GPU front end with the CPU's (`VVDEC_GPU_MIDER=1`).
- `GpuParse.*`: the GPU parse's inputs, built from a picture's slices, and the comparison with the CPU's parse.
- `GpuParseService.*`: one picture's parse on the GPU (`ParseSlot`), and the parse of the pictures ahead of their
  reconstruction (`ParseService`).
- `GpuFrontEnd.*`: the CU, TU and CTU records the CPU writes for the GPU front end, and the tables its shaders read.
- `GpuJobs.h`: the job structures shared by the CPU and the shaders.
- `GpuFilmGrain.*`: film grain synthesis on the output frames.
- `shaders/*.slang`: one file per pass (some with variants, see `cmake/modules/vvdecVulkanShaders.cmake`). The build
  embeds their SPIR-V in the generated `vvdec_gpu/GpuShaders.h`, which only `GpuContext.cpp` includes; the other
  files name shaders through `vvdec_gpu/GpuShaderIds.h`.
