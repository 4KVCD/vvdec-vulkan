# ab_steady.py <stream> <frames> <rounds> <exe names...>: steady-state fps (first frame to end of decoding), peak
# commit and working set, time to the first frame; the exes alternated, the GPU parse on, 16 threads.
import subprocess, re, statistics, os, sys
S = os.path.dirname(os.path.abspath(__file__))   # this directory: the exes under test, references, logs
PROC = S + r'\procstat\procstat.exe'
stream, frames, rounds, names = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
runs = {}
for r in range(rounds):
    for name in names:
        env = dict(os.environ)
        env['VVDEC_GPU_PARSE'] = '2'
        cmd = [PROC, '--mark', 'VVdeC,', '--mark', 'SizeInfo', '--mark', 'frames decoded @', '--', S + rf'\vvdecapp_{name}.exe', '-b', stream, '-f', str(frames), '-t', '16', '-v', '3', '--gpu', '1']
        out = subprocess.run(cmd, capture_output=True, text=True, env=env).stdout
        kv = {k: float(v) for k, v in re.findall(r'(\w+) (-?[\d.]+)', out[out.index('PROCSTAT'):])}
        fps = (frames - 1) / ((kv['mark2_ms'] - kv['mark1_ms']) / 1000)
        runs.setdefault(name, []).append((fps, kv['peak_commit_mb'], kv['peak_ws_mb'], kv['mark1_ms'] / 1000))
        print(f'{name} run {r + 1}: {fps:.1f} fps, commit {kv["peak_commit_mb"]:.0f} MB, ws {kv["peak_ws_mb"]:.0f} MB, first frame {kv["mark1_ms"] / 1000:.2f} s', flush=True)
for name in runs:
    cols = list(zip(*runs[name]))
    print(f'{name} median: {statistics.median(cols[0]):.1f} fps, commit {statistics.median(cols[1]):.0f} MB, ws {statistics.median(cols[2]):.0f} MB, first frame {statistics.median(cols[3]):.2f} s')
