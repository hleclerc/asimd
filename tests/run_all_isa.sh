#!/usr/bin/env bash
# EVERY TEST, AT EVERY x86 FEATURE LEVEL.
#
# `make` builds for -march=native, so on this machine it only ever exercises the AVX-512 paths.
# The interesting failures are elsewhere: intrinsics guarded by the wrong feature macro compile
# fine on a machine that has everything and not at all on the target that needs them, and a
# `require_at_least` that holds under AVX-512 says nothing about SSE2.
#
#   ./run_all_isa.sh            g++
#   ./run_all_isa.sh clang++
#
set -u
CXX=${1:-g++}
HERE=$( cd "$( dirname "$0" )" && pwd )
TMP=$( mktemp -d ); trap 'rm -rf "$TMP"' EXIT

command -v "$CXX" >/dev/null || { echo "$CXX not found"; exit 1; }

TESTS=( test_ops test_split test_selection test_x86_ops test_x86_dispatch )
# The last two rows are the same ISA with ASIMD_NO_COMPILER_VECTORS: that macro forces the array
# form of `values`, which is exactly what MSVC gets, since it has no `vector_size`. It is the only
# way to exercise the MSVC configuration without MSVC -- and what it checks is not portability but
# whether the dispatch table stands on its own, or has been leaning on gcc's vector arithmetic to
# cover its holes. `tests/no_vecext.sh` measures the same thing in instructions.
ISAS=( "-msse2|SSE2" "-msse4.2|SSE4.2" "-mavx|AVX" "-mavx2 -mfma|AVX2" "-march=native|native"
       "-msse2 -DASIMD_NO_COMPILER_VECTORS|SSE2, MSVC path"
       "-march=native -DASIMD_NO_COMPILER_VECTORS|native, MSVC path" )

rc=0
for isa in "${ISAS[@]}"; do
    flags=${isa%%|*}; label=${isa#*|}
    echo "===================== $label ($flags) ====================="
    for t in "${TESTS[@]}"; do
        if ! $CXX -std=c++20 -O2 $flags -Wall -I"$HERE/../src" -I"$HERE" \
                  "$HERE/$t.cpp" -o "$TMP/$t" 2> "$TMP/$t.err"; then
            echo "  BUILD FAILED  $t"
            grep -m3 "error:" "$TMP/$t.err" | sed 's/^/      /'
            rc=1; continue
        fi
        if grep -q "warning:" "$TMP/$t.err"; then
            echo "  warnings in $t:"; grep -m3 "warning:" "$TMP/$t.err" | sed 's/^/      /'
        fi
        "$TMP/$t" > "$TMP/$t.out" 2>&1 || rc=1
        grep -E "^$t|failures" "$TMP/$t.out" | sed 's/^/  /'
        grep -E "^  broken|^  FAIL" "$TMP/$t.out" | sed 's/^/  /'
    done
done

echo
[ $rc -eq 0 ] && echo "all feature levels: ok" || echo "SOMETHING FAILED"
exit $rc
