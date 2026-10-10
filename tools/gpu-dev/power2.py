# power2.py <1080p|4K|path.266> <frames> <runs> name=exe[,extra,args]...
# Each exe in turn, <runs> rounds (interleaved): procstat's RAPL energy over a decode of the first <frames> frames
# (16 threads, no MD5), in the steady state only -- from the first frame output to the end of decoding (start-up and
# teardown left out): fps, and average power:
#   package, CPU cores, the rest of the chip (GPU, memory controller, fabric), the package above the idle power;
#   plus the fps vvdecapp reports, CPU threads busy, and the frames output late (paced builds).
# Then each exe's medians. Extra args '@NAME=value' set environment variables.
import os
import re
import statistics
import subprocess
import sys

S = os.path.dirname(os.path.abspath(__file__))   # this directory: the exes under test, references, logs
PROC = S + r'\procstat\procstat.exe'
BENCH = os.path.join(os.environ['TEMP'], 'vvc-bench')

res, frames, runs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
stream = res if res.endswith('.266') else f'videoq_{res}.266'
specs = []
for a in sys.argv[4:]:
    name, rest = a.split('=', 1)
    parts = rest.split(',')
    specs.append((name, parts[0], parts[1:]))


def idle():
    out = subprocess.run([PROC, '--idle', '5', '--', 'cmd', '/c', 'exit'], capture_output=True, text=True).stdout
    m = re.search(r'IDLE package ([\d.]+) W cores ([\d.]+) W', out)
    return float(m.group(1)), float(m.group(2))


idle_pkg, idle_cores = idle()
print(f'idle before: package {idle_pkg:.2f} W, cores {idle_cores:.2f} W', flush=True)
cols = ['fps', 'steady_s', 'pkg_W', 'cores_W', 'rest_W', 'above_idle_W', 'mJ_frame', 'startup_s', 'late']
table = {name: [] for name, _, _ in specs}
for r in range(runs):
    for name, exe, extra in specs:
        env = dict(os.environ)
        for a in extra:
            if a.startswith('@'):
                k, v = a[1:].split('=', 1)
                env[k] = v
        args = [a for a in extra if not a.startswith('@')]
        cmd = [PROC, '--mark', 'VVdeC,', '--mark', 'SizeInfo', '--mark', 'frames decoded @', '--', exe, '-b', stream, '-f', str(frames), '-t', '16', '-v', '3'] + args
        out = subprocess.run(cmd, capture_output=True, text=True, cwd=BENCH, env=env).stdout
        m = re.search(r'decoded @ ([\d.]+) fps', out)
        late = re.search(r'(\d+) frames were output late', out)
        p = out[out.index('PROCSTAT'):]
        kv = {k: float(v) for k, v in re.findall(r'(\w+) (-?[\d.]+)', p)}
        if kv['exit'] != 0 or not m:
            print(f'{name} run {r + 1}: FAILED (exit {kv["exit"]})')
            continue
        dt = (kv['mark2_ms'] - kv['mark1_ms']) / 1000
        pkg = (kv['mark2_pkg_j'] - kv['mark1_pkg_j']) / dt
        cores = (kv['mark2_cores_j'] - kv['mark1_cores_j']) / dt
        fps = (frames - 1) / dt
        row = [fps, dt, pkg, cores, pkg - cores, pkg - idle_pkg, pkg / fps * 1000, kv['mark1_ms'] / 1000, float(late.group(1)) if late else 0.0]
        table[name].append(row)
        print(f'{res} {name:>8} run {r + 1}: ' + '  '.join(f'{c} {v:.1f}' for c, v in zip(cols, row)), flush=True)
idle_pkg2, idle_cores2 = idle()
print(f'idle after: package {idle_pkg2:.2f} W, cores {idle_cores2:.2f} W')
print(f'{res} medians:')
print(f'{"":>10}' + ''.join(f'{c:>14}' for c in cols))
for name in table:
    if table[name]:
        med = [statistics.median(col) for col in zip(*table[name])]
        print(f'{name:>10}' + ''.join(f'{v:>14.1f}' for v in med))
