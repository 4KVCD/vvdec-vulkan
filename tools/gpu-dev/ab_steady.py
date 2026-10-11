# ab_steady.py <stream> <frames> <rounds> <specs...>: steady-state fps (first frame to end of decoding), peak commit and
# working set, time to the first frame; the specs alternated, 16 threads, medians.
# A spec is a program name (vvdecapp_<name>.exe here) with an optional mode: <name> (the GPU parses: VVDEC_GPU_PARSE=2),
# <name>:hyb (the CPU parses, the GPU does the rest), <name>:cpu (the CPU decoder, --gpu 0).
# Each run's output is kept in ab_logs/<spec>_run<n>.log; a run that fails or lacks its marks is reported, not counted.
import os
import re
import statistics
import subprocess
import sys

S = os.path.dirname(os.path.abspath(__file__))   # this directory: the exes under test, references, logs
PROC = S + r'\procstat\procstat.exe'
LOGS = os.path.join(S, 'ab_logs')
stream, frames, rounds, specs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
os.makedirs(LOGS, exist_ok=True)
runs = {}
for r in range(rounds):
    for spec in specs:
        name, _, mode = spec.partition(':')
        env = dict(os.environ)
        env.pop('VVDEC_GPU_PARSE', None)
        if mode == '':
            env['VVDEC_GPU_PARSE'] = '2'
        gpu = '0' if mode == 'cpu' else '1'
        cmd = [PROC, '--mark', 'VVdeC,', '--mark', 'SizeInfo', '--mark', 'frames decoded @', '--', S + rf'\vvdecapp_{name}.exe', '-b', stream, '-f', str(frames), '-t', '16', '-v', '3', '--gpu', gpu]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, errors='replace')
        out = proc.stdout + proc.stderr
        log = os.path.join(LOGS, f'{spec.replace(":", "_")}_run{r + 1}.log')
        open(log, 'w').write(out)
        at = out.find('PROCSTAT')
        kv = {k: float(v) for k, v in re.findall(r'(\w+) (-?[\d.]+)', out[at:])} if at >= 0 else {}
        if kv.get('exit', 1) != 0 or kv.get('mark2_ms', 0) <= kv.get('mark1_ms', 0):
            print(f'{spec} run {r + 1}: FAILED (exit {kv.get("exit")}, marks {kv.get("mark1_ms")} {kv.get("mark2_ms")}): see {log}', flush=True)
            continue
        fps = (frames - 1) / ((kv['mark2_ms'] - kv['mark1_ms']) / 1000)
        runs.setdefault(spec, []).append((fps, kv['peak_commit_mb'], kv['peak_ws_mb'], kv['mark1_ms'] / 1000))
        print(f'{spec} run {r + 1}: {fps:.1f} fps, commit {kv["peak_commit_mb"]:.0f} MB, ws {kv["peak_ws_mb"]:.0f} MB, first frame {kv["mark1_ms"] / 1000:.2f} s', flush=True)
for spec in runs:
    cols = list(zip(*runs[spec]))
    print(f'{spec} median: {statistics.median(cols[0]):.1f} fps, commit {statistics.median(cols[1]):.0f} MB, ws {statistics.median(cols[2]):.0f} MB, first frame {statistics.median(cols[3]):.2f} s')
