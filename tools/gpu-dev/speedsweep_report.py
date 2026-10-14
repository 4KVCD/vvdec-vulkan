# speedsweep_report.py <speed_<name>.json> [top]: ranks speedsweep.py's results against the CPU decoder.
# Steady speed: a video's marginal milliseconds a frame ((whole decode - 1-frame decode) / (frames - 1)), for videos of
# at least 30 frames, in each GPU configuration (gpu: the whole decoder on the GPU; hyb: the CPU parses) against the
# CPU's on the same video; the slowest relative to the CPU are listed with their size. Startup: the time
# to the first frame (the worker's "first"), per configuration. A video far slower than the CPU, or than other videos
# of its size, uses something the GPU decoder handles badly.
import json
import os
import statistics
import sys

res = json.load(open(sys.argv[1]))
top = int(sys.argv[2]) if len(sys.argv) > 2 else 25
configs = [c for c in ('gpu', 'hyb') if any(c in e for e in res.values())]


def marginal(e):
    if not e or e.get('error') or not e.get('seconds') or e.get('one') is None or e['frames'] < 30:
        return None
    return 1000.0 * max(1e-4, e['seconds'] - e['one']) / (e['frames'] - 1)


errors = [(v, c, e[c]['error']) for v, e in res.items() for c in e if isinstance(e[c], dict) and e[c].get('error')]
for c in configs:
    rows = []
    for v, e in res.items():
        g, cpu = marginal(e.get(c)), marginal(e.get('cpu'))
        if g is not None and cpu is not None:
            rows.append((cpu / g, v, e['width'], e['height'], e[c]['frames'], g, cpu))
    rows.sort()
    if not rows:
        continue
    med = statistics.median(r[0] for r in rows)
    print(f'== steady speed, {c} against the CPU decoder: {len(rows)} videos of >= 30 frames, median {med:.2f}x')
    print(f'   {"video":28s} {"size":>11s} {"frames":>6s} {c + " ms":>8s} {"cpu ms":>8s} {"vs cpu":>7s}')
    for q, v, w, h, f, g, cpu in rows[:top]:
        print(f'   {v:28s} {w:5d}x{h:<5d} {f:6d} {g:8.2f} {cpu:8.2f} {q:6.2f}x')
    print()
for c in configs + ['cpu']:
    firsts = sorted(((e[c].get('first') or 0) * 1000, v) for v, e in res.items() if c in e and not e[c].get('error') and e[c].get('first'))
    if firsts:
        print(f'== time to the first frame, {c}: median {statistics.median(x for x, _ in firsts):.0f} ms; slowest: ' +
              ', '.join(f'{v} {x:.0f}' for x, v in firsts[::-1][:12]))
if errors:
    print(f'\nerrors: {len(errors)}')
    for v, c, e in errors:
        print(f'   {v} {c}: {e}')
