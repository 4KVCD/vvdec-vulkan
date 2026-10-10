# GPU path development tools

Checks and measurements for the Vulkan decoder: see `VULKAN_HANDOFF.md` (section 6) at the repository root.

- `fastcheck.py`: the check of every change (`--tier smoke`, ~2 minutes) and of milestones (`--tier full`).
- `check.sh`, `check_job.sh`, `make_crc_refs.sh`: the full tier's QA round and its CRC-32C references.
- `ab_steady.py`, `stats.py`, `power2.py`: speed (steady state), memory, CPU time, energy and power, builds alternated.
- `procstat/`: the process launcher the measurements use (wall and CPU time, memory peaks, RAPL energy, output marks).
- `probes/`: temporary measurement patches (GPU buffers by allocation site, CPU heap by block size).
- `vkext/`: lists a device's Vulkan extensions.
- Data: `refs/` (per-frame MD5s made by upstream vvdec), `tools_md5_2.txt`, `crc_refs.txt`, `conf_upstream.txt`,
  `check_times.txt`, `fastcheck_times.json`.

The programs under test go in this directory as `vvdecapp_<name>.exe` and `vvdecbatch_<name>.exe`; the test streams
are expected in `%TEMP%\vvc-bench\`.
