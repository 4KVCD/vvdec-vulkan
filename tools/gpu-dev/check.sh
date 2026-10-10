#!/bin/bash
# check.sh <name>: the check-in round of vvdecapp_<name>.exe (a build with vvdecapp's VVDEC_FASTHASH):
# 1. the 250 conformance streams upstream vvdec decodes (MD5 against the JVET's .yuv.md5) and the tool streams
#    (tools_md5_2.txt's MD5s and options), two decodes at a time, the longest first (check_times.txt: the seconds of
#    the round before);
# 2. the QA round, the output's CRC-32C (VVDEC_FASTHASH=1) against crc_refs.txt (make_crc_refs.sh: upstream's output,
#    hashed fast): VideoQ 1080p 4800 frames (16 threads) and 1200 (2 threads), VideoQ 4K 1200 (16) and 600 (2), The
#    Beekeeper 600, the films' clips, the stress streams (one at a time: two 8K decodes at once run out of the GPU's
#    memory); and the CPU path (VideoQ 1080p 1200 frames, --gpu 0) against upstream's MD5.
# A decode that logs an error or an exception fails. Out: <name>_check.txt -- the counts, every line that isn't a
# pass, the minutes taken.
# Not in the round (they decode the same streams again; for finding a failure's cause): the shadow sweeps
# (mider_sweep.sh, parse_sweep.sh), the thread counts between 2 and 16.
S="$(cd "$(dirname "$0")" && pwd)"   # this directory: the exes under test, references, logs
B="$(cygpath -u "$TEMP")/vvc-bench"
C="$B/conformance/fdis_r1"
name="$1"
E="$S/vvdecapp_$name.exe"
out="$S/${name}_check.txt"
work="$S/check_${name}_work"
t0=$(date +%s)
rm -rf "$work" && mkdir -p "$work"

# 1. the conformance and tool streams
{
  awk '$2 == "PASS" {print $1}' "$S/conf_upstream.txt" | while read -r p; do
    echo "$p $C/$p/$p.bit $(cut -c1-32 "$C/$p/$p.yuv.md5")"
  done
  while read -r t md5 opts; do
    echo "tool_$t $B/tools/$t.266 $md5 $opts"
  done < "$S/tools_md5_2.txt"
} | sed 's/ *$//' > "$work/jobs.txt"   # (xargs -L continues a line that ends in a blank onto the next)
awk 'NR == FNR { secs[$1] = $2; next } { print ( $1 in secs ? secs[$1] : 1 ), $0 }' "$S/check_times.txt" "$work/jobs.txt" 2> /dev/null \
  | sort -k1,1 -rn | cut -d' ' -f2- > "$work/order.txt"
[ -s "$work/order.txt" ] || cp "$work/jobs.txt" "$work/order.txt"
xargs -P 2 -L 1 bash "$S/check_job.sh" "$E" "$work" < "$work/order.txt"
cut -d' ' -f1 "$work/jobs.txt" | while read -r j; do cat "$work/$j.txt" 2> /dev/null || echo "$j MISSING"; done > "$S/check_${name}_streams.txt"
awk '{ s = $3; sub( "s$", "", s ); print $1, s }' "$S/check_${name}_streams.txt" > "$S/check_times.txt"
t1=$(date +%s)

# 2. the QA round
ref() { grep "^$1 " "$S/crc_refs.txt" | cut -d' ' -f2; }
qa() {   # label, the reference's label, stream, frames (all), vvdecapp args...
  local label=$1 r; r=$(ref "$2"); local stream=$3 frames=$4; shift 4
  local fr=""; [ "$frames" != "all" ] && fr="-f $frames"
  if [ -z "$r" ]; then echo "FAIL $label: no reference (make_crc_refs.sh)"; return; fi
  local log; log=$(VVDEC_FASTHASH=1 timeout 600 "$E" -b "$stream" $fr -md5 "${r}000000000000000000000000" -v 3 "$@" 2>&1)
  local rc=$?
  local fps; fps=$(echo "$log" | grep -o "decoded @ [0-9.]* fps" | tail -1)
  if [ $rc -eq 0 ] && ! echo "$log" | grep -q "mismatch" && ! echo "$log" | grep -qi "exception\|ERROR"; then
    echo "PASS $label: $fps"
  else
    echo "FAIL $label (rc=$rc): $fps $(echo "$log" | grep -i -m1 "mismatch\|error\|exception" | tr -d '\r' | cut -c1-120)"
  fi
}
{
  qa vq1080_4800 vq1080_4800 "$B/videoq_1080p.266" 4800 --gpu 1 -t 16
  qa vq1080_1200_t2 vq1080_1200 "$B/videoq_1080p.266" 1200 --gpu 1 -t 2
  qa vq4k_1200 vq4k_1200 "$B/videoq_4K.266" 1200 --gpu 1 -t 16
  qa vq4k_600_t2 vq4k_600 "$B/videoq_4K.266" 600 --gpu 1 -t 2
  qa bee600 bee600 "$B/beekeeper_medium_4K.266" 600 --gpu 1 -t 16
  while read -r m md5; do qa "movie_$m" "movie_$m" "$B/$m.266" all --gpu 1 -t 16; done < "$B/movie_md5.txt"
  for i in $(seq 1 720); do mkdir "$S/stress.lock" 2> /dev/null && break; sleep 5; done
  while read -r st md5; do qa "stress_$st" "stress_$st" "$B/stress/$st.266" all --gpu 1 -t 16; done < "$B/stress/ref_md5.txt"
  rmdir "$S/stress.lock"
  # the CPU path, against upstream's MD5 itself
  log=$("$E" -b "$B/videoq_1080p.266" -f 1200 -md5 43129838872aad31ab351a48acbff9fa --gpu 0 -t 16 -v 3 2>&1)
  rc=$?
  if [ $rc -eq 0 ] && ! echo "$log" | grep -q "mismatch" && ! echo "$log" | grep -qi "exception\|ERROR"; then echo "PASS cpu_path"; else echo "FAIL cpu_path (rc=$rc)"; fi
} > "$S/check_${name}_qa.txt" 2>&1
t2=$(date +%s)

{
  f="$S/check_${name}_streams.txt"
  echo "$name: conformance $(grep -v '^tool_' "$f" | grep -c ' PASS ') PASS of $(grep -vc '^tool_' "$f"); tools $(grep '^tool_' "$f" | grep -c ' PASS ') PASS of $(grep -c '^tool_' "$f"); QA $(grep -c '^PASS' "$S/check_${name}_qa.txt") PASS of $(wc -l < "$S/check_${name}_qa.txt")"
  grep -v ' PASS ' "$f"
  grep -v '^PASS' "$S/check_${name}_qa.txt"
  echo "minutes: streams $(( ( t1 - t0 + 30 ) / 60 )), QA $(( ( t2 - t1 + 30 ) / 60 )), total $(( ( t2 - t0 + 30 ) / 60 ))"
} > "$out"
cat "$out"
