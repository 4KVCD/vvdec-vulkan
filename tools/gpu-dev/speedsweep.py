# speedsweep.py <vvdecbatch exe> <name> [--only REGEX] [--configs gpu,cpu[,hyb]] [--repeat N]
# The decoder's speed on every video of the check (the conformance streams upstream decodes, the tool streams, the
# films but VideoQ), one decode at a time, to find the videos where the GPU decoder is abnormally slow: a sign that a
# feature they use is not well optimized. Each configuration has its own vvdecbatch worker (gpu: the whole decoder on
# the GPU, VVDEC_GPU_PARSE=2; hyb: the CPU parses, the GPU does the rest; cpu: the CPU decoder, gpu 0); the workers
# take one job at a time from this script, so only one decode runs at once.
#
# A video's time per frame is its marginal one: (its whole decode's seconds - a 1-frame decode's) / (frames - 1), so
# that opening the decoder and filling its pipeline do not count (short conformance streams would be all that). The
# whole decode runs --repeat times (default 2 when it takes under 3 s, else 1) and the quickest counts.
# Out: speed_<name>.json (per video and configuration: frames, seconds, 1-frame seconds, width, height) beside this
# script; speedsweep_report.py ranks it.
import json
import os
import re
import subprocess
import sys
import time

S = os.path.dirname(os.path.abspath(__file__))
B = os.path.join(os.environ['TEMP'], 'vvc-bench')
C = os.path.join(B, 'conformance', 'fdis_r1')
REFS = os.path.join(S, 'refs')

args = sys.argv[1:]
exe, name = args[0], args[1]
opts = {'--only': '', '--configs': 'gpu,cpu', '--repeat': '0'}
i = 2
while i < len(args):
    opts[args[i]] = args[i + 1]
    i += 2
only = re.compile(opts['--only']) if opts['--only'] else None
configs = opts['--configs'].split(',')
repeat = int(opts['--repeat'])
OUT = os.path.join(S, f'speed_{name}.json')
FIELD_STREAMS = {'FIELD_A_4', 'FIELD_B_2'}   # (vvdecbatch does not decode field pictures as frames)
FILMS = {   # name -> frames at most (the films are long: a share of them is enough)
    'bee_slower_cut': None, 'tg_slower_cut': None, 'beekeeper_slower_4K': 300, 'beekeeper_medium_4K': 300,
    'bee_1608': 300, 'topgun_slower_1080p': 600, 'topgun_medium_1080p': 600, 'topgun_1080p': 600, 'conan_1080p': 600,
    'fg_tg4k_600': 300, 'fg_8bit_x40': None, 'isp_ai_4k_x8': None, 'mip_ai_4k_x8': None,
}


def opl_size(path):
    """(width, height, frames) of an .opl list"""
    w = h = n = 0
    for line in open(path):
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 5:
            w, h, n = max(w, int(parts[2])), max(h, int(parts[3])), n + 1
    return w, h, n


def app_size(bit):
    """(width, height) from upstream vvdecapp's SizeInfo line"""
    out = subprocess.run([os.path.join(S, 'cpub_master.exe'), '-b', bit, '-f', '1', '-t', '1', '-v', '4'], capture_output=True, text=True,
                         errors='replace', timeout=120).stdout
    m = re.search(r'SizeInfo: (\d+)x(\d+)', out)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def videos():
    """name -> (bitstream, frames at most or None, film grain, width, height, frames in the stream or None)"""
    out = {}
    for line in open(os.path.join(S, 'conf_upstream.txt')):
        parts = line.split()
        if len(parts) >= 2 and parts[1] == 'PASS' and parts[0] not in FIELD_STREAMS:
            p = parts[0]
            w, h, n = opl_size(os.path.join(C, p, p + '.opl'))
            out[p] = (os.path.join(C, p, p + '.bit'), None, 0, w, h, n)
    for line in open(os.path.join(S, 'tools_md5_2.txt')):
        parts = line.split()
        if len(parts) >= 2:
            t = 'tool_' + parts[0]
            w, h, n = opl_size(os.path.join(REFS, t + '.opl'))
            out[t] = (os.path.join(B, 'tools', parts[0] + '.266'), None, 1 if '-fg' in parts[2:] else 0, w, h, n)
    for f, most in FILMS.items():
        bit = os.path.join(B, f + '.266')
        if not os.path.exists(bit):
            continue
        opl = os.path.join(REFS, f + '.opl')
        if os.path.exists(opl):
            w, h, n = opl_size(opl)
        else:
            (w, h), n = app_size(bit), None
        out[f] = (bit, most, 1 if f.startswith('fg_') else 0, w, h, n)
    return out


class Worker:
    """A vvdecbatch process taking jobs on stdin, one at a time"""

    def __init__(self, config):
        env = dict(os.environ)
        # the compiled pipelines of the program (fastcheck.py's and ab_steady.py's file for it): a decode must not
        # compile them
        program = re.sub(r'^vvdecbatch_?', '', os.path.splitext(os.path.basename(exe))[0]) or name
        env['VVDEC_GPU_CACHE'] = os.path.join(os.environ['LOCALAPPDATA'], 'vvdec', 'tools', program + '.bin')
        if config == 'gpu':
            env['VVDEC_GPU_PARSE'] = '2'
        env['VVDECBATCH_NOHASH'] = '1'   # (an MD5 a frame would cap a 4K decode near 27 fps)
        self.gpu = 0 if config == 'cpu' else 1
        self.p = subprocess.Popen([exe, '-'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env,
                                  bufsize=1)
        self.jobs = 0
        self.parts = {}

    def decode(self, bit, frames, fg, timeout=900):
        """(frames decoded, seconds) or (frames, error text)"""
        self.p.stdin.write(f'{bit} {frames or 0} 16 {self.gpu} {fg}\n')
        self.p.stdin.flush()
        index, got, t0 = self.jobs, 0, time.time()
        self.jobs += 1
        while True:
            line = self.p.stdout.readline()
            if not line:
                return got, 'worker ended'
            if line.startswith(f'job {index} frame'):
                got += 1
            elif line.startswith(f'job {index} end'):
                rest = line.split('end', 1)[1].strip()
                if not rest.startswith('OK'):
                    return got, rest
                words = rest.split()
                # (newer workers add: open <s> first <s> close <s>)
                self.parts = {words[k]: float(words[k + 1]) for k in range(2, len(words) - 1, 2)}
                return got, float(words[1])
            if time.time() - t0 > timeout:
                return got, 'timeout'

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=30)
        except Exception:
            self.p.kill()


def main():
    vids = videos()
    names = sorted(n for n in vids if not only or only.search(n))
    try:
        results = json.load(open(OUT))
    except (OSError, ValueError):
        results = {}
    workers = {c: Worker(c) for c in configs}
    # a first decode in each worker, not counted: the device, the pipelines and the memory pools come up
    warm = vids.get('bee_slower_cut') or next(iter(vids.values()))
    for wk in workers.values():
        wk.decode(warm[0], 8, 0)
    t_start = time.time()
    for k, v in enumerate(names):
        bit, most, fg, w, h, n = vids[v]
        entry = results.setdefault(v, {'width': w, 'height': h})
        for c in configs:
            best, frames, err, parts = None, 0, None, {}
            for r in range(max(1, repeat) if repeat else 2):
                got, sec = workers[c].decode(bit, most, fg)
                if isinstance(sec, str):
                    err = sec
                    break
                frames = got
                if best is None or sec < best:
                    best, parts = sec, dict(workers[c].parts)
                if not repeat and sec >= 3.0:
                    break
            one = workers[c].decode(bit, 1, fg)[1] if err is None else None
            entry[c] = {'frames': frames, 'seconds': best, 'one': one if not isinstance(one, str) else None, 'error': err, **parts}
        json.dump(results, open(OUT, 'w'), indent=0, sort_keys=True)
        line = '  '.join(f"{c} {entry[c]['frames']}f {entry[c]['seconds'] or 0:.2f}s" + (f" ERR {entry[c]['error']}" if entry[c]['error'] else '')
                         for c in configs)
        print(f'[{k + 1}/{len(names)} {time.time() - t_start:.0f}s] {v} {w}x{h}: {line}', flush=True)
    for wk in workers.values():
        wk.close()


main()
