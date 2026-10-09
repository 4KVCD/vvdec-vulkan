# VulkanLib: reconstructing VVC pictures on the GPU

With `vvdecParams::gpu` set (`vvdecapp --gpu 1`), vvdec reconstructs pictures with Vulkan compute shaders. The
output is bit-exact with the CPU decoder. The CPU still parses the bitstream (CABAC), handles the parameter sets,
picture and slice headers and reference picture lists, and records and submits the GPU work. This document is the
overview; each source file and each shader starts with a description of its own part.

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
parameters, job lists and edges with the CPU's (`GpuRecon::compareMider`). It is a debugging aid.

## A picture's journey (path 1)

**Tasks** (`DecLibRecon`, on the decoder's thread pool):

| task | runs when | does |
|---|---|---|
| CTU tasks | the picture is parsed | write the CU and TU records |
| `filterParamsTask` | the picture is parsed | works out the SAO, ALF and deblocking parameters (`GpuRecon::prepareFilters`) |
| `queueGpuTask` | its CTU tasks are done and every reference picture's GPU work is queued (`gpuRefReady`) | records and submits the picture's GPU work (`GpuRecon::startPicture`), then unlocks the picture's `gpuRefReady` |
| `computeDoneTask` | the picture's GPU work is queued | waits for the GPU's compute work (`GpuRecon::endCompute`), then unlocks `gpuComputeDone`: the decoder instance may take its next picture |
| `finishGpuTask` | `gpuComputeDone` | waits for the picture's download to host memory (`GpuRecon::endDownload`) |

**Queues and submissions** (`GpuRecon::startPicture`):

| queue | submission | contents | waits for |
|---|---|---|---|
| async compute (one per decoder instance) | front end, part 1 | motion, colocated motion, job lists | the first parts of the picture's colocated pictures (the timeline semaphore) |
| async compute | front end, part 2 | intra order, deblocking parameters, deblocking edges | front end part 1 (same queue) |
| compute | first part | uploads of reference pictures the CPU made, residuals, inter prediction (DMVR writes its refined motion into the colocated motion) | front end part 1; then signals the timeline semaphore |
| compute | rest | intra prediction, LMCS's inverse mapping, deblocking, SAO, ALF and CC-ALF | front end part 2 |
| copy | download | the finished picture to host memory | the rest |

The next picture's front end reads the refined motion through TMVP, which is why it waits for its colocated
pictures' first parts.

**Pass order inside the picture**: residuals (`trafo.slang`) -> inter prediction (`inter.slang`) -> intra prediction
in dependency waves (`intra.slang`) -> LMCS inverse mapping (`lmcs.slang`) -> deblocking (`deblock.slang`) -> SAO
(`sao.slang`) -> ALF and CC-ALF (`alf.slang`). Film grain (`filmgrain.slang`, `GpuFilmGrain`) is applied to the
output frames, not to the reference pictures.

## Glossary

- **host plane / twin**: every picture plane lives in host memory that the GPU can reach (`hostPictureAllocator`),
  where vvdec's CPU code reads and writes it. Its **twin** is a copy in the GPU's own memory, where the GPU
  reconstructs the picture and keeps it as a reference picture.
- **scratch planes**: picture-sized GPU memory taken from a pool for SAO's and ALF's output; after ALF they become
  the picture's twin.
- **first part / rest**: the two command buffers of a picture's work on the compute queue (see above).
- **front end**: everything between parsing and reconstruction: motion derivation, deblocking parameters, job
  lists, intra order, deblocking edges.
- **records**: the per-CU, per-TU and per-CTU words the CPU's parse writes for the GPU front end (`GpuFrontEnd.h`).
- **jobs**: the work items of a reconstruction pass (`GpuJobs.h`): `TrafoJob`, `InterJob`, `IntraJob`.
- **waves**: groups of intra jobs that may run at the same time; a job's wave is one more than the highest wave
  of the blocks it reads from.
- **intra bands**: intra blocks split into strips of 256 samples, the unit of work of `intra.slang`.
- **early rows** ("band" in some names): with the CPU front end, the top half of a picture's CTU rows, whose
  residuals and inter blocks are submitted before the CPU finishes the rest (`GpuRecon::startBand`).
- **deferred predictions**: inter predictions kept aside for an intra job to finish (LMCS chroma scaling, CIIP).

## Environment variables

| variable | effect |
|---|---|
| `VVDEC_GPU_DEVICE=<n>` | use the Vulkan device with index n (default: the first discrete GPU, else the first integrated one) |
| `VVDEC_GPU_PROFILE=1` | print the GPU time of each pass and each shader's registers and memory at exit |
| `VVDEC_GPU_ISA=<dir>` | with `VVDEC_GPU_PROFILE=1` on AMD: write each shader's machine code to `<dir>/<name>.isa` |
| `VVDEC_GPU_FRONT=0` | path 2: the CPU builds the GPU's jobs |
| `VVDEC_GPU_MIDER=1` | path 2 plus the GPU front end, compared with the CPU's (debugging) |
| `VVDEC_GPU_ASYNC=0` | run the front end on the compute queue instead of an async compute queue |
| `VVDEC_FASTHASH=1` | (vvdecapp) `-md5` hashes the output with CRC-32C instead of MD5, for fast checks |

## Files

- `GpuContext.*`: the Vulkan device, its queues, memory allocation and the shaders' pipelines.
- `GpuRecon.*`: records and submits a picture's GPU work, from the uploads to the download.
- `GpuFrontEnd.*`: the CU, TU and CTU records the CPU writes for the GPU front end, and the comparisons with the
  CPU's results used by `VVDEC_GPU_MIDER=1`.
- `GpuJobs.h`: the job structures shared by the CPU and the shaders.
- `GpuFilmGrain.*`: film grain synthesis on the output frames.
- `shaders/*.slang`: one file per pass (some with variants, see `cmake/modules/vvdecVulkanShaders.cmake`).
