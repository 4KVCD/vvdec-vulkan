#!/bin/bash
# make_crc_refs.sh <exe with VVDEC_FASTHASH>: check.sh's QA references, once -- for each QA decode, the output's CRC-32C
# (VVDEC_FASTHASH=1) by the exe's CPU path (--gpu 0), kept in crc_refs.txt only when the same CPU path's MD5 matches
# the decode's known MD5 (upstream vvdec's): the references are upstream's output, hashed fast.
S="$(cd "$(dirname "$0")" && pwd)"   # this directory: the exes under test, references, logs
B="$(cygpath -u "$TEMP")/vvc-bench"
E="$1"
refs="$S/crc_refs.txt"
touch "$refs"
zero=00000000000000000000000000000000
# label, stream, frames (all: the whole stream), the upstream MD5
items=( "vq1080_4800 $B/videoq_1080p.266 4800 f28db72a2a30c7ff1013c0e261e14b77"
        "vq1080_1200 $B/videoq_1080p.266 1200 43129838872aad31ab351a48acbff9fa"
        "vq4k_1200 $B/videoq_4K.266 1200 a868a35c72f1772e2660eca4ac89ea7e"
        "bee600 $B/beekeeper_medium_4K.266 600 e4b773bdc80ef9af1171e8418ec28f09" )
while read -r name md5; do items+=( "movie_$name $B/$name.266 all $md5" ); done < "$B/movie_md5.txt"
while read -r name md5; do items+=( "stress_$name $B/stress/$name.266 all $md5" ); done < "$B/stress/ref_md5.txt"
# (VideoQ 4K 600 frames: no upstream MD5 of its own -- its CRC from the CPU path whose 1200 frames matched)
for item in "${items[@]}" "vq4k_600 $B/videoq_4K.266 600 -"; do
  read -r label stream frames md5 <<< "$item"
  grep -q "^$label " "$refs" && continue
  fr=""; [ "$frames" != "all" ] && fr="-f $frames"
  if [ "$md5" != "-" ]; then
    out=$("$E" -b "$stream" $fr -md5 "$md5" --gpu 0 -t 16 -v 3 2>&1)
    if [ $? -ne 0 ] || echo "$out" | grep -q "mismatch"; then
      echo "NO REFERENCE $label: the CPU path's MD5 differs from upstream's"
      continue
    fi
  fi
  crc=$(VVDEC_FASTHASH=1 "$E" -b "$stream" $fr -md5 $zero --gpu 0 -t 16 -v 3 2>&1 | grep -o "!= [0-9a-f]\{8\}" | head -1 | cut -c4-)
  [ -n "$crc" ] && echo "$label $crc" >> "$refs" && echo "$label $crc"
done
