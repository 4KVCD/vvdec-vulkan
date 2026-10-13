# fastcheck.py <vvdecapp exe> <name> [--tier smoke|wide|full] [--modes default,parse[,cpufe][,pcheck]] [--lanes N] [--budget T]
#              [--only REGEX] [--batch <vvdecbatch exe>] [--refs]
# The Vulkan decoder's check, with the decodes side by side. Every decode is checked frame by frame: each output
# frame's per-plane MD5 against the JVET's per-frame list (<stream>.opl, conformance) or a list made once by upstream
# vvdec (refs/<job>.opl; --refs makes them with cpub_master.exe, the upstream build, checking its whole output against
# the known MD5 in the same run). So a decode cut to its first frames is checked exactly, frame by frame.
#
# The decodes run in vvdecbatch workers (one process decodes job after job, the Vulkan device and pipelines up the
# whole time: a vvdecapp process per stream spends ~1.2 s setting them up and down), each worker taking a job as it
# finishes the one before. vvdecapp runs what the workers can't: the whole-output CRC runs of the QA round, the CPU
# path, the field-coded streams.
#
# Tiers:
#   smoke (each change, about a minute): one stream per conformance family (its quickest), cut to ~3 s of decoding
#         (at most 17 frames); every tool stream's first 6 frames; the slower-preset cuts (Beekeeper 4K 98 frames,
#         Top Gun 1080p 114 frames: most VVC tools on real content);
#   wide  (now and then): every conformance stream and tool stream, each cut to ~6 s of decoding; the slower-preset
#         cuts, the long slower-preset streams' first frames (Beekeeper 240, Top Gun 600); VideoQ cut;
#   full  (milestones): everything whole (the long slower-preset streams: Beekeeper 600 frames, Top Gun 1200), plus
#         check.sh's QA round (VideoQ 4800 / 1200 frames at 16 and 2 threads,
#         The Beekeeper, the films, the stress streams; CRC-32C against crc_refs.txt) and the CPU path.
# Modes: default (the CPU parses), parse (VVDEC_GPU_PARSE=2: the GPU parses), cpufe (VVDEC_GPU_FRONT=0: the CPU's front
# end, for changes to it; only when asked for), pcheck (VVDEC_GPU_PARSE=1: the CPU parses, and the GPU parses each
# picture too for a comparison with the CPU's records and levels, for changes to the parse's host side; only when asked
# for). Each mode has its own workers. The workers log errors only, so pcheck catches crashes and wrong output; the
# comparison's differences show in a vvdecapp run's summary ("GPU: parse (VVDEC_GPU_PARSE) of ...").
# Scheduling: longest first (times remembered in fastcheck_times.json), at most --lanes decodes at a time, within a
# budget of GPU memory tokens (an 8K decode takes the whole budget). A decode that fails is decoded again alone with
# vvdecapp afterwards, to tell a real failure from collateral damage (a GPU fault in one process resets the GPU work
# of the others): both results are reported.
# Out: <name>_fast.txt (the counts, a line per decode that did not pass, the minutes) and the same on stdout.
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time

S = os.path.dirname(os.path.abspath(__file__))   # this directory: the exes under test, references, logs
B = os.path.join(os.environ['TEMP'], 'vvc-bench')
C = os.path.join(B, 'conformance', 'fdis_r1')
REFS = os.path.join(S, 'refs')
UPSTREAM = os.path.join(S, 'cpub_master.exe')
TIMES = os.path.join(S, 'fastcheck_times.json')
LOGS = os.path.join(S, 'fastcheck_logs')   # the vvdecapp decodes' output (the last run's)
os.makedirs(LOGS, exist_ok=True)

args = sys.argv[1:]
exe, name = args[0], args[1]
opts = {'--tier': 'smoke', '--modes': 'default,parse', '--lanes': '8', '--budget': '8', '--only': '', '--batch': ''}
flags = set()
i = 2
while i < len(args):
    if args[i] in opts:
        opts[args[i]] = args[i + 1]
        i += 2
    else:
        flags.add(args[i])
        i += 1
tier = opts['--tier']
modes = opts['--modes'].split(',')
lanes = int(opts['--lanes'])
budget = int(opts['--budget'])
only = re.compile(opts['--only']) if opts['--only'] else None
batch_exe = opts['--batch'] or os.path.join(os.path.dirname(exe), os.path.basename(exe).replace('vvdecapp', 'vvdecbatch'))
# The decoder's modes (see Modes above)
MODE_ENV = {'default': {}, 'parse': {'VVDEC_GPU_PARSE': '2'}, 'cpufe': {'VVDEC_GPU_FRONT': '0'}, 'pcheck': {'VVDEC_GPU_PARSE': '1'}}
# The compiled pipelines' file of this program (ab_steady.py uses the same): the decoder starts its file anew when the
# shaders differ, so programs sharing one would compile every shader again whenever another ran in between.
CACHE = os.path.join(os.environ['LOCALAPPDATA'], 'vvdec', 'tools', name + '.bin')
os.makedirs(os.path.dirname(CACHE), exist_ok=True)
os.environ['VVDEC_GPU_CACHE'] = CACHE
FIELD_STREAMS = {'FIELD_A_4', 'FIELD_B_2'}   # (vvdecbatch does not hash field pictures: these use vvdecapp's MD5)
HASH_RE = re.compile(r'frame (\d+)\s+\[MD5:([0-9a-f,]+)\]')


# --------------------------------------------------------------------------------------------------------------------
# The streams and their per-frame references
# --------------------------------------------------------------------------------------------------------------------

def read_opl(path):
    """The frames of an .opl list: (width, height, (md5 per plane...)) in output order."""
    frames = []
    for line in open(path):
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 5:
            frames.append((int(parts[2]), int(parts[3]), tuple(parts[4:])))
    return frames


def tokens_of(frames):
    """The GPU memory tokens of a decode, from its largest picture."""
    area = max(w * h for w, h, _ in frames)
    return 8 if area > 3840 * 2160 * 2 else 3 if area > 1920 * 1088 * 2 else 2 if area > 960 * 544 * 2 else 1


def conformance_streams():
    """The conformance streams upstream vvdec decodes: name -> (bitstream, frames, expected MD5 of the whole output)."""
    out = {}
    for line in open(os.path.join(S, 'conf_upstream.txt')):
        parts = line.split()
        if len(parts) >= 2 and parts[1] == 'PASS':
            p = parts[0]
            md5 = open(os.path.join(C, p, p + '.yuv.md5')).read()[:32]
            out[p] = (os.path.join(C, p, p + '.bit'), read_opl(os.path.join(C, p, p + '.opl')), md5)
    return out


def family(stream):
    return re.sub(r'_[A-Z]_\d+$', '', stream)


def tool_streams():
    """The tool streams: name -> (bitstream, extra args, whole-output MD5 of the CPU path)."""
    out = {}
    for line in open(os.path.join(S, 'tools_md5_2.txt')):
        parts = line.split()
        if len(parts) >= 2:
            out['tool_' + parts[0]] = (os.path.join(B, 'tools', parts[0] + '.266'), parts[2:], parts[1])
    return out


# The QA streams with per-frame references (refs/<name>.opl): name -> (bitstream, frames for the references or None
# for all, MD5 of upstream's output for those frames). The slower-preset encodes use most VVC tools (affine, GPM,
# SbTMVP, BDOF, CIIP, MIP, ISP, LFNST...); the cuts are 3-4 s from the middle of the films (their first frames are
# black), made with ffmpeg -ss 100 -t 3 (4 for Top Gun) -c copy -bsf:v vvc_mp4toannexb.
QA_REF_STREAMS = {
    'videoq_1080p': (os.path.join(B, 'videoq_1080p.266'), 1200, '43129838872aad31ab351a48acbff9fa'),
    'videoq_4K': (os.path.join(B, 'videoq_4K.266'), 1200, 'a868a35c72f1772e2660eca4ac89ea7e'),
    'beekeeper_slower_4K': (os.path.join(B, 'beekeeper_slower_4K.266'), 600, 'a41ee6ab48b3841549b3c598f37ed478'),
    'topgun_slower_1080p': (os.path.join(B, 'topgun_slower_1080p.266'), 1200, '7e641f2d4270a8e7fbe95e3eaeedf066'),
    'bee_slower_cut': (os.path.join(B, 'bee_slower_cut.266'), None, 'c391b05c7d31ce408d26eeb4707acb47'),
    'tg_slower_cut': (os.path.join(B, 'tg_slower_cut.266'), None, 'c1c68a15e8d15a3be2a36c2657b1cbfa'),
}


def make_refs():
    """refs/<name>.opl for the tool streams and the QA streams, made by upstream vvdec (CPU) with -dph 11, its whole
    output checked against the known MD5 in the same run."""
    os.makedirs(REFS, exist_ok=True)
    jobs = []
    for job, (bit, extra, md5) in tool_streams().items():
        jobs.append((job, [UPSTREAM, '-b', bit, '-dph', '11', '-md5', md5, '-v', '3'] + extra))
    for job, (bit, frames, md5) in QA_REF_STREAMS.items():
        cut = ['-f', str(frames)] if frames else []
        jobs.append((job, [UPSTREAM, '-b', bit] + cut + ['-dph', '11', '-md5', md5, '-v', '3']))
    running = []
    failed = 0
    while jobs or running:
        while jobs and len(running) < 4:
            job, cmd = jobs.pop(0)
            log = open(os.path.join(LOGS, job + '.log'), 'w')
            running.append((job, subprocess.Popen(cmd, cwd=B, stdout=log, stderr=subprocess.STDOUT), log))
        for entry in list(running):
            job, proc, log = entry
            if proc.poll() is None:
                continue
            running.remove(entry)
            log.close()
            out = open(log.name, errors='replace').read()
            hashes = HASH_RE.findall(out)
            size = re.search(r'SizeInfo: (\d+)x(\d+)', out)
            w, h = (int(size.group(1)), int(size.group(2))) if size else (0, 0)
            if proc.returncode != 0 or 'mismatch' in out or not hashes:
                print(f'refs {job}: FAILED (rc {proc.returncode})', flush=True)
                failed += 1
                continue
            with open(os.path.join(REFS, job + '.opl'), 'w') as f:
                for idx, md5s in hashes:
                    f.write(f'  0, {int(idx):7d}, {w:4d}, {h:4d},{md5s}\n')
            print(f'refs {job}: {len(hashes)} frames', flush=True)
        time.sleep(0.05)
    return failed


# --------------------------------------------------------------------------------------------------------------------
# The jobs
# --------------------------------------------------------------------------------------------------------------------

class Job:
    """A decode: in a vvdecbatch worker (frames checked against refs), or with vvdecapp (app: its whole output's MD5
    or CRC, or a check of its own)."""

    def __init__(self, jid, bit, mode, frames, refs, threads=16, gpu=1, film_grain=0, tokens=None):
        self.jid, self.bit, self.mode, self.frames, self.refs = jid, bit, mode, frames, refs
        self.threads, self.gpu, self.film_grain = threads, gpu, film_grain
        self.tokens = tokens if tokens is not None else tokens_of(refs)
        self.app = None          # vvdecapp: (args, env additions) -- then refs are unused
        self.timeout = 300
        self.result = None
        self.estimate = 3.0

    def batch_line(self):
        return f'{self.bit} {self.frames or 0} {self.threads} {self.gpu} {self.film_grain}\n'

    def check_frames(self, lines, end):
        """PASS, or what went wrong, from a worker's lines of this job and its end line."""
        got = [m.group(2) for m in (HASH_RE.search(l) for l in lines) if m]
        want = self.refs if self.frames is None else self.refs[:self.frames]
        for k, (md5s, (_, _, ref)) in enumerate(zip(got, want)):
            planes = md5s.split(',')
            n = min(len(planes), len(ref))
            if tuple(planes[:n]) != ref[:n]:
                return f'MISMATCH at frame {k} of {len(want)}'
        if not end.startswith('OK'):
            return end[:120]
        if len(got) != len(want):
            return f'FRAMES {len(got)} of {len(want)}'
        return 'PASS'

    def check_app(self, rc, out):
        if rc is None:
            return 'TIMEOUT'
        first = next((l for l in out.splitlines() if re.search(r'error|exception|not supported|mismatch', l, re.I)), '')
        if 'mismatch' in out:
            return 'MISMATCH ' + first[:100]
        if rc != 0:
            return f'ERROR rc={rc} ' + first[:100]
        return 'PASSWARN ' + first[:100] if first else 'PASS'


def app_job(jid, bit, mode, tokens, args, env=None, timeout=600):
    j = Job(jid, bit, mode, None, [(0, 0, ())], tokens=tokens)
    j.app = (args, env or {})
    j.timeout = timeout
    return j


def build_jobs():
    conf = conformance_streams()
    tools = tool_streams()
    try:
        times = json.load(open(TIMES))
    except (OSError, ValueError):
        times = {}
    # The seconds a frame of each conformance stream takes, for cutting and choosing: a mode's own, as this script last
    # decoded the stream (its first frames, side by side), or else the GPU-parse mode's average over the whole stream
    # in check.sh's last run (a vvdecapp process's own ~1.2 s taken off)
    whole_run = {}
    try:
        for line in open(os.path.join(S, 'check_times.txt')):
            parts = line.split()
            if len(parts) == 2 and parts[0] in conf:
                whole_run[parts[0]] = max(0.0, float(parts[1]) - 1.2) / max(1, len(conf[parts[0]][1]))
    except OSError:
        pass

    def per_frame(p, mode):
        last = times.get(f'{p}/{mode}')
        if isinstance(last, list) and last[1]:
            return last[0] / last[1]
        return whole_run.get(p, 0.05)

    def cut(p, mode, seconds, most):
        frames = conf[p][1]
        keep = max(2, min(most, int(seconds / max(per_frame(p, mode), 1e-3))))
        return keep if keep < len(frames) else None

    jobs = []
    for mode in modes:
        conf_jobs = []
        if tier == 'smoke':
            # the quickest stream of each family
            best = {}
            for p in conf:
                cost = per_frame(p, mode) * min(17, len(conf[p][1]))
                if family(p) not in best or cost < best[family(p)][0]:
                    best[family(p)] = (cost, p)
            conf_jobs = [(p, cut(p, mode, 2.0, 17)) for _, p in best.values()]
        else:
            conf_jobs = [(p, cut(p, mode, 6.0, 10 ** 6) if tier == 'wide' else None) for p in conf]
        for p, frames in conf_jobs:
            bit, refs, md5 = conf[p]
            if p in FIELD_STREAMS:
                jobs.append(app_job(f'{p}/{mode}', bit, mode, tokens_of(refs), ['-md5', md5]))
            else:
                jobs.append(Job(f'{p}/{mode}', bit, mode, frames, refs))
        tool_frames = 6 if tier == 'smoke' else 48 if tier == 'wide' else None
        for t, (bit, extra, _) in tools.items():
            refs = read_opl(os.path.join(REFS, t + '.opl'))
            fg = 1 if '-fg' in extra else 0
            jobs.append(Job(f'{t}/{mode}', bit, mode, tool_frames if tool_frames and tool_frames < len(refs) else None, refs, film_grain=fg))
        # the slower-preset cuts whole in every tier; the long streams' first frames in the wide tier, and in the full one
        # all their references but the last 8: upstream's vvdecapp made them stopping after that many frames, which flushes
        # the pictures decoded so far, so its last frames are other pictures than a decode that goes on outputs (Beekeeper
        # slower 4K, -f 600: frame 598 differs); vvdecbatch goes on (and the whole Beekeeper, 4305 frames, outlasts the
        # timeout beside 7 other decodes)
        for s in ('bee_slower_cut', 'tg_slower_cut'):
            jobs.append(Job(f'{s}/{mode}', QA_REF_STREAMS[s][0], mode, None, read_opl(os.path.join(REFS, s + '.opl'))))
        if tier != 'smoke':
            for s in ('beekeeper_slower_4K', 'topgun_slower_1080p'):
                frames = (240 if '4K' in s else 600) if tier == 'wide' else QA_REF_STREAMS[s][1] - 8
                jobs.append(Job(f'{s}/{mode}', QA_REF_STREAMS[s][0], mode, frames, read_opl(os.path.join(REFS, s + '.opl'))))
        if tier == 'wide':
            jobs.append(Job(f'vq1080/{mode}', QA_REF_STREAMS['videoq_1080p'][0], mode, 600, read_opl(os.path.join(REFS, 'videoq_1080p.opl'))))
            jobs.append(Job(f'vq4k/{mode}', QA_REF_STREAMS['videoq_4K'][0], mode, 240, read_opl(os.path.join(REFS, 'videoq_4K.opl'))))
        elif tier == 'full':
            jobs += qa_jobs(mode)
    if tier == 'full':
        jobs.append(cpu_path_job())   # the CPU path, against upstream's MD5 of the whole output
    for j in jobs:
        last = times.get(j.jid, 3.0)
        j.estimate = last[0] if isinstance(last, list) else last
    if only:
        jobs = [j for j in jobs if only.search(j.jid)]
    return jobs


def cpu_path_job():
    j = app_job('cpu_path', os.path.join(B, 'videoq_1080p.266'), 'default', 2, ['-f', '1200', '-md5', '43129838872aad31ab351a48acbff9fa', '-t', '16'])
    j.gpu = 0
    return j


def qa_jobs(mode):
    """check.sh's QA round: whole runs against the CRC-32C references (crc_refs.txt), with vvdecapp."""
    refs = {}
    for line in open(os.path.join(S, 'crc_refs.txt')):
        parts = line.split()
        if len(parts) == 2:
            refs[parts[0]] = parts[1]

    def crc(label, ref, stream, frames, threads, tokens):
        a = ['-md5', refs[ref] + '0' * 24, '-t', str(threads)] + (['-f', str(frames)] if frames else [])
        return app_job(f'{label}/{mode}', stream, mode, tokens, a, {'VVDEC_FASTHASH': '1'})

    q = os.path.join(B, 'videoq_1080p.266')
    k = os.path.join(B, 'videoq_4K.266')
    jobs = [crc('vq1080_4800', 'vq1080_4800', q, 4800, 16, 2), crc('vq1080_1200_t2', 'vq1080_1200', q, 1200, 2, 2),
            crc('vq4k_1200', 'vq4k_1200', k, 1200, 16, 3), crc('vq4k_600_t2', 'vq4k_600', k, 600, 2, 3),
            crc('bee600', 'bee600', os.path.join(B, 'beekeeper_medium_4K.266'), 600, 16, 3)]
    for line in open(os.path.join(B, 'movie_md5.txt')):
        m = line.split()[0]
        jobs.append(crc('movie_' + m, 'movie_' + m, os.path.join(B, m + '.266'), None, 16, 3))
    for line in open(os.path.join(B, 'stress', 'ref_md5.txt')):
        st = line.split()[0]
        big = '8k' in st or 'mixed' in st or 'switch' in st
        jobs.append(crc('stress_' + st, 'stress_' + st, os.path.join(B, 'stress', st + '.266'), None, 16, 8 if big else 3))
    return jobs


# --------------------------------------------------------------------------------------------------------------------
# Running them
# --------------------------------------------------------------------------------------------------------------------

class Worker:
    """A vvdecbatch process of one mode, decoding the jobs it is handed one at a time; a thread queues its lines."""

    def __init__(self, mode):
        env = dict(os.environ)
        env.update(MODE_ENV[mode])
        self.mode = mode
        self.proc = subprocess.Popen([batch_exe, '-'], cwd=B, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, errors='replace', bufsize=1)
        self.lines = queue.Queue()
        self.job = None
        self.got = []
        threading.Thread(target=self.read, daemon=True).start()

    def read(self):
        for line in self.proc.stdout:
            self.lines.put(line.rstrip('\n'))
        self.lines.put(None)   # (the process ended)

    def give(self, job):
        self.job, self.got, job.start = job, [], time.time()
        self.proc.stdin.write(job.batch_line())
        self.proc.stdin.flush()

    def poll(self):
        """The job's result once its end line came (or the process died), else None."""
        while True:
            try:
                line = self.lines.get_nowait()
            except queue.Empty:
                if time.time() - self.job.start > self.job.timeout:
                    self.proc.kill()
                    return self.keep('TIMEOUT')
                return None
            if line is None:
                return self.keep(f'WORKER DIED (rc {self.proc.wait()})')
            m = re.match(r'job \d+ end (.*)', line)
            if m:
                self.got.append(line)
                return self.keep(self.job.check_frames(self.got[:-1], m.group(1)))
            if line.startswith('job '):
                self.got.append(line)
            elif line.strip():
                self.got.append(line)   # (the library's error messages count for the check's lines too)

    def keep(self, result):
        """Returns result; when the job did not pass, first writes its lines to fastcheck_logs/<job>_batch.log: the
        decode alone afterwards writes a log of its own, and a failure that comes only beside other decodes may not
        come again."""
        if result != 'PASS':
            with open(os.path.join(LOGS, self.job.jid.replace('/', '_') + '_batch.log'), 'w') as f:
                f.write('\n'.join(self.got + [result]) + '\n')
        return result

    def close(self):
        try:
            self.proc.stdin.close()
        except OSError:
            pass


def start_app(job):
    env = dict(os.environ)
    env.update(MODE_ENV[job.mode])
    env.update(job.app[1])
    cmd = [exe, '-b', job.bit, '--gpu', str(job.gpu), '-v', '2'] + job.app[0]
    job.log = open(os.path.join(LOGS, job.jid.replace('/', '_') + '.log'), 'w')
    job.proc = subprocess.Popen(cmd, cwd=B, env=env, stdout=job.log, stderr=subprocess.STDOUT)
    job.start = time.time()


# A GPU fault -- a lost device, a Vulkan call failing, a decode hung or dead -- stops the whole run: the other decodes
# share the GPU, and each retry would hang it again. Several GPU resets in a row once took the laptop's graphics
# driver down with them (a bugcheck, 2026-10-10).
gpu_fault = None


def looks_like_gpu_fault(text):
    t = text.lower()
    return any(k in t for k in ('device lost', 'device_lost', '(-4)', 'vkallocatememory failed', 'waiting for the gpu',
                                'vkqueuesubmit failed', 'timeout', 'worker died'))


def run(jobs, parallel, use_workers=True):
    """Runs the jobs, longest first, at most `parallel` at a time within the token budget; sets each job's result.
    Stops everything at the first GPU fault (see gpu_fault)."""
    global gpu_fault
    pending = sorted(jobs, key=lambda j: -j.estimate)
    idle = {mode: [] for mode in MODE_ENV}
    busy = []      # workers with a job
    apps = []      # vvdecapp jobs running
    used = 0

    def finish(job, result, text=''):
        global gpu_fault
        nonlocal used
        job.result = result
        job.seconds = time.time() - job.start
        used -= job.tokens
        if result != 'PASS':
            print(f'  {job.jid}: {result} ({job.seconds:.1f} s)', flush=True)
            if gpu_fault is None and looks_like_gpu_fault(result + '\n' + text):
                gpu_fault = f'{job.jid}: {result}'

    while pending or busy or apps:
        if gpu_fault is not None:
            # a GPU fault: every decode stopped at once, nothing more started (each retry would hang the GPU again)
            for w in busy:
                w.proc.kill()
            for j in apps:
                j.proc.kill()
                j.proc.wait()
                j.log.close()
            busy, apps, pending = [], [], []
            break
        # hand out jobs while there is room
        for j in list(pending):
            if len(busy) + len(apps) >= parallel:
                break
            if used + j.tokens > budget and (busy or apps):
                continue
            pending.remove(j)
            used += j.tokens
            if j.app is not None or not use_workers:
                if j.app is None:
                    j.app = (['-dph', '11'], {'VVDEC_OUTFRAMES': str(j.frames)} if j.frames else {})
                start_app(j)
                apps.append(j)
            else:
                w = idle[j.mode].pop() if idle[j.mode] else Worker(j.mode)
                w.give(j)
                busy.append(w)
        for w in list(busy):
            result = w.poll()
            if result is None:
                continue
            busy.remove(w)
            finish(w.job, result, '\n'.join(w.got))
            if w.proc.poll() is None:
                idle[w.mode].append(w)
        for j in list(apps):
            rc = j.proc.poll()
            if rc is None and time.time() - j.start < j.timeout:
                continue
            if rc is None:
                j.proc.kill()
                j.proc.wait()
            j.log.close()
            out = open(j.log.name, errors='replace').read()
            apps.remove(j)
            if j.app[0][:2] == ['-dph', '11']:
                # (a worker's job, decoded alone with vvdecapp: its frames checked the same way)
                lines = [l for l in out.splitlines() if 'picHash' in l]
                result = j.check_frames(lines, 'OK' if rc == 0 else f'ERROR rc={rc}')
            else:
                result = j.check_app(rc, out)
            finish(j, result, out)
        time.sleep(0.01)
    for ws in idle.values():
        for w in ws:
            w.close()


def warm_up():
    """A short decode per mode first, in a worker and in vvdecapp, so that the side-by-side decodes find the shaders
    compiled (the driver compiles them once per program)."""
    small = os.path.join(C, 'AMVR_A_3', 'AMVR_A_3.bit')
    for mode in modes:
        env = dict(os.environ)
        env.update(MODE_ENV[mode])
        subprocess.run([batch_exe, '-'], cwd=B, env=env, input=f'{QA_REF_STREAMS["videoq_1080p"][0]} 16 16 1 0\n{small} 2 16 1 0\n',
                       text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run([exe, '-b', small, '-f', '2', '--gpu', '1', '-v', '0'], cwd=B, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    if '--refs' in flags:
        sys.exit(make_refs())
    t0 = time.time()
    jobs = build_jobs()
    print(f'{name}: {tier}, modes {",".join(modes)}: {len(jobs)} decodes, {lanes} at a time', flush=True)
    warm_up()
    t1 = time.time()
    run(jobs, lanes)
    print(f'  side by side: {(time.time() - t1) / 60:.2f} min (warm-up {(t1 - t0):.0f} s)', flush=True)
    if gpu_fault is not None:
        print(f'{name}: GPU FAULT, everything stopped: {gpu_fault}. Nothing retried: fix the cause before more GPU work '
              '(repeated GPU resets can take the graphics driver down).', flush=True)
        sys.exit(2)
    failed = [j for j in jobs if j.result != 'PASS']
    alone = {}
    if failed:
        # each again, alone, with vvdecapp
        print(f'  again alone: {len(failed)}', flush=True)
        retry = []
        for j in failed:
            r = Job(j.jid, j.bit, j.mode, j.frames, j.refs, j.threads, j.gpu, j.film_grain, j.tokens)
            r.app, r.timeout, r.estimate = j.app if j.app and j.app[0][:2] != ['-dph', '11'] else None, j.timeout, 0
            retry.append(r)
        run(retry, 1, use_workers=False)
        alone = {r.jid: r.result for r in retry}
    # the times, for the next run's order
    try:
        times = json.load(open(TIMES))
    except (OSError, ValueError):
        times = {}
    for j in jobs:
        if hasattr(j, 'seconds'):
            # [seconds, frames decoded]: the next run's estimate and cut
            times[j.jid] = [round(j.seconds, 2), j.frames or (len(j.refs) if j.app is None else 0)]
    json.dump(times, open(TIMES, 'w'), indent=0, sort_keys=True)
    passed = sum(1 for j in jobs if j.result == 'PASS')
    timed = sorted((j for j in jobs if hasattr(j, 'seconds')), key=lambda j: -j.seconds)
    print('  longest: ' + ', '.join(f'{j.jid} {j.seconds:.1f} s' for j in timed[:6]), flush=True)
    lines = [f'{name}: {tier}, modes {",".join(modes)}: {passed} PASS of {len(jobs)} decodes; {(time.time() - t0) / 60:.1f} min']
    for j in failed:
        lines.append(f'{j.jid}: {j.result}; alone: {alone.get(j.jid)}')
    open(os.path.join(S, f'{name}_fast.txt'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    sys.exit(1 if failed else 0)


main()
