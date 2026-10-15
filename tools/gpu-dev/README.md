# GPU path development tools

Checks and measurements for the Vulkan decoder.

- `fastcheck.py`: the check of every change (`--tier smoke`, ~2 minutes) and of milestones (`--tier full`).
- `check.sh`, `check_job.sh`, `make_crc_refs.sh`: the full tier's QA round and its CRC-32C references.
- `ab_steady.py`, `stats.py`, `power2.py`: speed (steady state), memory, CPU time, energy and power, builds alternated.
- `speedsweep.py`, `speedsweep_report.py`: every video's speed against the CPU decoder, one decode at a time, ranked.
- `procstat/`: the process launcher the measurements use (wall and CPU time, memory peaks, RAPL energy, output marks).
- `probes/`: temporary measurement patches (GPU buffers by allocation site, CPU heap by block size).
- `vkext/`: lists a device's Vulkan extensions.
- `uiprobe/`: what the desktop sees of the GPU during a decode: a small submission on the graphics queue every 1/60 s,
  timed from submission to completion (median, percentiles, and how many waited past a frame).
- Data: `refs/` (per-frame MD5s made by upstream vvdec), `tools_md5_2.txt`, `crc_refs.txt`, `conf_upstream.txt`,
  `check_times.txt`, `fastcheck_times.json`.

The programs under test go in this directory as `vvdecapp_<name>.exe` and `vvdecbatch_<name>.exe`; the test streams
are expected in `%TEMP%\vvc-bench\`.
