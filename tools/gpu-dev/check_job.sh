#!/bin/bash
# check_job.sh <exe> <work dir> <name> <bitstream> <expected MD5> [extra vvdecapp args...]: one decode of check.sh's
# conformance and tool streams with the full-output MD5 check; its line into <work dir>/<name>.txt: name, PASS /
# PASSWARN (MD5 right, but an error or exception logged on the way) / MISMATCH / ERROR rc / TIMEOUT, seconds, the
# first error line
exe="$1"; work="$2"; name="$3"; bit="$4"; md5="$5"
shift 5
start=$(date +%s.%N)
log=$(timeout 300 "$exe" -b "$bit" -md5 "$md5" -v 2 --gpu 1 "$@" 2>&1)
rc=$?
secs=$(awk -v a="$start" -v b="$(date +%s.%N)" 'BEGIN { printf "%.1f", b - a }')
first=$(echo "$log" | grep -i -m1 "error\|exception\|not supported\|mismatch" | tr -d '\r' | cut -c1-140)
if [ $rc -eq 124 ]; then
  status=TIMEOUT
elif echo "$log" | grep -q "MD5 mismatch"; then
  status=MISMATCH
elif [ $rc -eq 0 ] && [ -n "$first" ]; then
  status=PASSWARN
elif [ $rc -eq 0 ]; then
  status=PASS
else
  status="ERROR rc=$rc"
fi
echo "$name $status ${secs}s $first" > "$work/$name.txt"
