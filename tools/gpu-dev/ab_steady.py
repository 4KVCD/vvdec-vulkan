# ab_steady.py <stream> <frames> <rounds> <specs...>: steady-state fps (first frame to end of decoding), the package's
# power and energy per frame over the same span (RAPL), peak commit and working set, time to the first frame; the
# specs alternated, their order reversed every other round (the package's power often drifts through a run: a spec
# always run after another would always get less), 16 threads, medians.
# A spec is a program name (vvdecapp_<name>.exe here) with an optional mode: <name> (the GPU parses: VVDEC_GPU_PARSE=2),
# <name>:hyb (the CPU parses, the GPU does the rest), <name>:cpu (the CPU decoder, --gpu 0); and environment variables
# of its own after the name: <name>+VAR=value+VAR2=value[:mode].
# Each run's output is kept in ab_logs/<stream>_<spec>_run<n>.log; a run that fails or lacks its marks stops the timing.
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


def cache_file(name):
    """The compiled pipelines' file of program <name>. Each program keeps its own: the decoder starts its file anew
    when the shaders differ, so two builds sharing one would compile every shader again at each run (~15 s, while the
    parse runs ahead and inflates the steady-state fps)."""
    folder = os.path.join(os.environ['LOCALAPPDATA'], 'vvdec', 'tools')
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, name + '.bin')


def program(spec):
    """The program name of a spec: before its mode and its variables."""
    return spec.partition(':')[0].split('+')[0]


def environment(spec):
    """The environment of a spec's runs: its mode's variables, its own variables (name+VAR=value+...), and its
    program's pipeline file."""
    name, _, mode = spec.partition(':')
    name, *variables = name.split('+')
    env = dict(os.environ)
    env.pop('VVDEC_GPU_PARSE', None)
    env['VVDEC_GPU_CACHE'] = cache_file(name)
    if mode == '':
        env['VVDEC_GPU_PARSE'] = '2'
    for v in variables:
        key, _, value = v.partition('=')
        env[key] = value
    return env


# A short decode of each spec first, so that no timed run compiles shaders (the stream's own variants included)
for spec in specs:
    name, mode = program(spec), spec.partition(':')[2]
    subprocess.run([os.path.join(S, f'vvdecapp_{name}.exe'), '-b', stream, '-f', '8', '-t', '16', '-v', '0', '--gpu', '0' if mode == 'cpu' else '1'],
                   env=environment(spec), capture_output=True)
runs = {}
for r in range(rounds):
    for spec in (specs if r % 2 == 0 else specs[::-1]):
        name, mode = program(spec), spec.partition(':')[2]
        env = environment(spec)
        gpu = '0' if mode == 'cpu' else '1'
        cmd = [PROC, '--mark', 'VVdeC,', '--mark', 'SizeInfo', '--mark', 'frames decoded @', '--', S + rf'\vvdecapp_{name}.exe', '-b', stream, '-f', str(frames), '-t', '16', '-v', '3', '--gpu', gpu]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, errors='replace')
        out = proc.stdout + proc.stderr
        log = os.path.join(LOGS, f'{os.path.splitext(os.path.basename(stream))[0]}_{re.sub(r"[^A-Za-z0-9_.=-]", "_", spec)}_run{r + 1}.log')
        open(log, 'w').write(out)
        at = out.find('PROCSTAT')
        kv = {k: float(v) for k, v in re.findall(r'(\w+) (-?[\d.]+)', out[at:])} if at >= 0 else {}
        if kv.get('exit', 1) != 0 or kv.get('mark2_ms', 0) <= kv.get('mark1_ms', 0):
            # A failed decode may be a GPU fault: nothing more runs (repeated GPU resets can take the driver down).
            print(f'{spec} run {r + 1}: FAILED (exit {kv.get("exit")}, marks {kv.get("mark1_ms")} {kv.get("mark2_ms")}): see {log}; '
                  'stopped', flush=True)
            sys.exit(2)
        seconds = (kv['mark2_ms'] - kv['mark1_ms']) / 1000
        fps     = (frames - 1) / seconds
        # the package's (RAPL) power and energy per frame over the same steady-state span
        joules  = kv.get('mark2_pkg_j', -1) - kv.get('mark1_pkg_j', -1)
        watts   = joules / seconds if kv.get('mark1_pkg_j', -1) >= 0 else float('nan')
        mj      = 1000 * joules / (frames - 1) if kv.get('mark1_pkg_j', -1) >= 0 else float('nan')
        runs.setdefault(spec, []).append((fps, kv['peak_commit_mb'], kv['peak_ws_mb'], kv['mark1_ms'] / 1000, watts, mj))
        print(f'{spec} run {r + 1}: {fps:.1f} fps, package {watts:.1f} W ({mj:.0f} mJ a frame), commit {kv["peak_commit_mb"]:.0f} MB, '
              f'ws {kv["peak_ws_mb"]:.0f} MB, first frame {kv["mark1_ms"] / 1000:.2f} s', flush=True)
for spec in runs:
    cols = list(zip(*runs[spec]))
    print(f'{spec} median: {statistics.median(cols[0]):.1f} fps, package {statistics.median(cols[4]):.1f} W ({statistics.median(cols[5]):.0f} mJ a frame), '
          f'commit {statistics.median(cols[1]):.0f} MB, ws {statistics.median(cols[2]):.0f} MB, first frame {statistics.median(cols[3]):.2f} s')
