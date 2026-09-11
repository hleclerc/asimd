#!/usr/bin/env bash
# THE COVERAGE MAP. What a runtime test cannot tell you: which (operation, type, width, ISA)
# combinations do not COMPILE at all. A missing register form is not a slow path here -- the
# generic fallback goes through `data.split`, which a register impl does not have, so the whole
# cell is a hard error. That is invisible until someone writes the line.
#
#   ./compile_matrix.sh              g++, the four feature levels of its own target
#   ./compile_matrix.sh clang++      idem, with another compiler
#
# THE LEVELS FOLLOW THE COMPILER'S TARGET, read off `-dumpmachine` rather than off `uname`, for
# the same reason as in `run_all_isa.sh`: `-msse2` handed to an aarch64 compiler is not a lower
# feature level, it is a hard error, and every cell would come out `--` for a reason that has
# nothing to do with coverage.
#
# Each cell is one translation unit. `ok` = compiles, `--` = does not.

set -u
CXX=${1:-g++}
SRC=$( cd "$( dirname "$0" )/../src" && pwd )
TMP=$( mktemp -d )
trap 'rm -rf "$TMP"' EXIT

command -v "$CXX" >/dev/null || { echo "$CXX not found"; exit 1; }

TARGET=$( "$CXX" -dumpmachine 2>/dev/null || echo unknown )
case "$TARGET" in
    aarch64*|arm64*|arm-*|armv*)
        ISAS=( "-march=armv8-a|ARMv8-A" "-march=armv8.2-a+fp16|ARMv8.2+fp16"
               "-march=armv8.4-a+dotprod|ARMv8.4+dot" "-march=native|native" ) ;;
    *)
        ISAS=( "-msse2|SSE2" "-mavx|AVX" "-mavx2 -mfma|AVX2" "-march=native|native" ) ;;
esac

# SI16 AND SI8 ARE IN THE LIST NOW. They were not, and they are the types where a coverage hole
# is most likely: the x86 backend registers almost nothing for them below AVX-512BW, and the ARM
# one registers everything, because Advanced SIMD is uniform across lane widths in a way SSE is
# not. A map that stops at 32-bit lanes cannot see either fact.
TYPES=( FP32 FP64 SI32 PI32 SI64 PI64 SI16 SI8 )
# name|expression, with `a` and `b` two SimdVec and `V` the vector type
OPS=(
  "add|a + b"
  "sub|a - b"
  "mul|a * b"
  "div|a / b"
  "and|a & b"
  "shl|a << b"          # n/a on the floating point types: `<<` on a float is meant to be rejected
  "min|min( a, b )"
  "max|max( a, b )"
  "sum|V( a.sum() )"
  "fma|fma( a, b, b )"
  "cmp|V( T( any( a > b ) ) )"
  "bits|V( T( to_bits( gt( a, b ) ) ) )"
  "sel|select( gt( a, b ), a, b )"
  "perm|permute( a, SimdVec<SI32,NL>::iota( 0 ) )"
  "iota|V::iota( T( 0 ) )"
  "iotam|V::iota( T( 0 ), T( 2 ) )"
  "gath|V::gather( (const T *)nullptr, SimdVec<SI32,NL>::iota( 0 ) )"
)

job() { # $1 flags  $2 type  $3 expr  $4 outfile
  cat > "$4.cpp" <<EOF
#include <asimd/asimd.h>
using namespace asimd;
using T = $2;
static constexpr int NL = SimdSize<T>::value;
using V = SimdVec<T,NL>;
V f( V a, V b ) { return $3; }
EOF
  if $CXX -std=c++20 -O1 $1 -I"$SRC" -c "$4.cpp" -o "$4.o" >/dev/null 2>&1
    then echo ok > "$4.res"; else echo -- > "$4.res"; fi
}
export -f job; export CXX SRC

n=0
for isa in "${ISAS[@]}"; do for ty in "${TYPES[@]}"; do for op in "${OPS[@]}"; do
  printf '%s\0%s\0%s\0%s\0' "${isa%%|*}" "$ty" "${op#*|}" "$TMP/c$n"
  n=$(( n + 1 ))
done; done; done > "$TMP/jobs"

# `nproc` is GNU coreutils; macOS has neither it nor a drop-in, so ask sysctl and fall back.
NPROC=$( nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4 )

echo "$CXX targets $TARGET: ${n} translation units on $NPROC jobs..." >&2
xargs -0 -n4 -P "$NPROC" bash -c 'job "$0" "$1" "$2" "$3"' < "$TMP/jobs"

n=0; nb_bad=0
for isa in "${ISAS[@]}"; do
  printf '\n===== %s  (%s) =====\n' "${isa#*|}" "${isa%%|*}"
  printf '%-6s' ""; for op in "${OPS[@]}"; do printf '%-6s' "${op%%|*}"; done; printf '\n'
  for ty in "${TYPES[@]}"; do
    printf '%-6s' "$ty"
    for op in "${OPS[@]}"; do
      r=$( cat "$TMP/c$n.res" 2>/dev/null || echo '?' )
      # `<<` on a floating point type is meant to be rejected, not supported.
      if [ "${op%%|*}" = shl ] && { [ "$ty" = FP32 ] || [ "$ty" = FP64 ]; }; then r='n/a'
      elif [ "$r" = '--' ]; then nb_bad=$(( nb_bad + 1 )); fi
      printf '%-6s' "  $r"
      n=$(( n + 1 ))
    done; printf '\n'
  done
done

printf '\n%d of %d cells do not compile.\n' "$nb_bad" "$n"
