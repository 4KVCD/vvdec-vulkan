# stats.py <1080p|4K|path.266> <frames> <runs> name=exe[,extra,args]...
# Each exe in turn, <runs> rounds (interleaved): procstat's measures of a decode of the first <frames> frames of the
# VideoQ stream (16 threads, no MD5), one line a run, then each exe's medians:
#   fps, CPU ms a frame (user + kernel), threads busy on average, peak working set and private commit (MB),
#   page faults a frame, package energy a frame (mJ; cores, and the rest of the chip: GPU, memory, fabric),
#   the energy above the idle power a frame, the time to the decoder's start-up line and to the first frame (ms).
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
print(f'idle before: package {idle_pkg:.2f} W, cores {idle_cores:.2f} W')
cols = ['fps', 'cpu_ms_f', 'threads', 'ws_mb', 'commit_mb', 'pf_f', 'pkg_mj_f', 'core_mj_f', 'rest_mj_f', 'above_idle_mj_f', 'open_ms',
        'first_ms']
table = {name: [] for name, _, _ in specs}
for r in range(runs):
    for name, exe, extra in specs:
        env = dict(os.environ)   # extra args '@NAME=value': environment variables
        for a in extra:
            if a.startswith('@'):
                k, v = a[1:].split('=', 1)
                env[k] = v
        args = [a for a in extra if not a.startswith('@')]
        cmd = [PROC, '--mark', 'VVdeC,', '--mark', 'SizeInfo', '--', exe, '-b', stream, '-f', str(frames), '-t', '16', '-v', '3'] + args
        out = subprocess.run(cmd, capture_output=True, text=True, cwd=BENCH, env=env).stdout
        m = re.search(r'decoded @ ([\d.]+) fps', out)
        p = out[out.index('PROCSTAT'):]
        kv = {k: float(v) for k, v in re.findall(r'(\w+) (-?[\d.]+)', p)}
        if kv['exit'] != 0 or not m:
            print(f'{name} run {r + 1}: FAILED (exit {kv["exit"]})')
            continue
        n = frames
        wall = kv['wall_ms'] / 1000
        cpu = kv['user_ms'] + kv['kernel_ms']
        row = [float(m.group(1)), cpu / n, cpu / kv['wall_ms'], kv['peak_ws_mb'], kv['peak_commit_mb'], kv['page_faults'] / n,
               kv['pkg_j'] * 1000 / n, kv['cores_j'] * 1000 / n, (kv['pkg_j'] - kv['cores_j']) * 1000 / n,
               (kv['pkg_j'] - idle_pkg * wall) * 1000 / n, kv['mark0_ms'], kv['mark1_ms']]
        table[name].append(row)
        print(f'{res} {name:>8} run {r + 1}: ' + '  '.join(f'{c} {v:.1f}' for c, v in zip(cols, row)), flush=True)
idle_pkg2, idle_cores2 = idle()
print(f'idle after: package {idle_pkg2:.2f} W, cores {idle_cores2:.2f} W')
print(f'{res} medians:')
print(f'{"":>10}' + ''.join(f'{c:>16}' for c in cols))
for name in table:
    if table[name]:
        med = [statistics.median(col) for col in zip(*table[name])]
        print(f'{name:>10}' + ''.join(f'{v:>16.1f}' for v in med))
