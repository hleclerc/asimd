#!/usr/bin/env bash
# EVERY TEST, AT EVERY FEATURE LEVEL OF WHATEVER ARCHITECTURE THE COMPILER TARGETS.
#
# `make` builds for -march=native, so it only ever exercises what this machine happens to have.
# The interesting failures are elsewhere: an intrinsic guarded by the wrong feature macro compiles
# fine on a machine that has everything and not at all on the target that needs it, and a
# `require_at_least` that holds under AVX-512 says nothing about SSE2.
#
#   ./run_all_isa.sh            g++
#   ./run_all_isa.sh clang++
#
# THE LEVEL LIST FOLLOWS THE TARGET, not the host, and it is read off the compiler rather than off
# `uname`: the compiler may be a cross compiler, and `-msse2` handed to an aarch64 one is not a
# lower feature level but a hard error. This mattered the moment there were two backends.
set -u
CXX=${1:-g++}
HERE=$( cd "$( dirname "$0" )" && pwd )
TMP=$( mktemp -d ); trap 'rm -rf "$TMP"' EXIT

command -v "$CXX" >/dev/null || { echo "$CXX not found"; exit 1; }

TESTS=( test_ops test_split test_selection test_x86_ops test_x86_dispatch
        test_arm_ops test_arm_dispatch )

# The `MSVC path` rows are the same ISA with ASIMD_NO_COMPILER_VECTORS: that macro forces the
# array form of `values`, which is exactly what MSVC gets, since it has no `vector_size`. It is
# the only way to exercise the MSVC configuration without MSVC -- and what it checks is not
# portability but whether the dispatch table stands on its own, or has been leaning on the
# compiler's vector arithmetic to cover its holes. It matters more on ARM, not less: MSVC's ARM64
# target is a real one. `tests/no_vecext.sh` measures the same thing in instructions.
TARGET=$( "$CXX" -dumpmachine 2>/dev/null || echo unknown )
case "$TARGET" in
    aarch64*|arm64*|arm-*|armv*)
        # ARM. Advanced SIMD has been 128 bits since 2005, so these levels do not widen the
        # register -- they add instructions at the same width. `armv8-a` is the A64 baseline;
        # +fp16 and +dotprod are the extensions `ArmCpuFeatures.h` declares markers for.
        #
        # THERE IS NO ARMv7 ROW, and it is not an oversight: an ARMv7 build needs a cross
        # compiler, which is exactly what a CI job is for and not what a developer has. The
        # 32-bit lattice is exercised instead through `ArmCpu<64,NEON,FMA>`, a type -- see the
        # `V7` arch in `test_arm_ops.cpp`, which runs the whole grid over it on every host.
        ISAS=( "-march=armv8-a|ARMv8-A" "-march=armv8.2-a+fp16|ARMv8.2 +fp16"
               "-march=armv8.4-a+dotprod|ARMv8.4 +dotprod" "-march=native|native"
               "-march=armv8-a -DASIMD_NO_COMPILER_VECTORS|ARMv8-A, MSVC path"
               "-march=native -DASIMD_NO_COMPILER_VECTORS|native, MSVC path" )
        ;;
    *)
        # THE `-mavx` MSVC-PATH ROW IS NOT REDUNDANT, and its absence hid a compile error for as
        # long as the MSVC job existed. AVX gives the 256-bit register IMPLS; AVX2 gives the
        # 256-bit integer INSTRUCTIONS. Between the two sits a configuration where a register impl
        # exists with no register form of `add`/`sub`, so the generic form runs over an array --
        # and that is MSVC's `/arch:AVX` precisely. SSE2 does not reach it (128-bit int add/sub are
        # registered) and neither does native (the 256-bit ones are).
        ISAS=( "-msse2|SSE2" "-msse4.2|SSE4.2" "-mavx|AVX" "-mavx2 -mfma|AVX2" "-march=native|native"
               "-msse2 -DASIMD_NO_COMPILER_VECTORS|SSE2, MSVC path"
               "-mavx -DASIMD_NO_COMPILER_VECTORS|AVX, MSVC path (= /arch:AVX)"
               "-march=native -DASIMD_NO_COMPILER_VECTORS|native, MSVC path" )
        ;;
esac
echo "$CXX targets $TARGET"

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
        # EVERY VERDICT LINE, not two of them. This was `^  broken|^  FAIL`, which silently
        # dropped `check.h`'s `XPASS` -- a known-broken assertion that starts passing, i.e. the
        # one line that says "go promote this to a CHECK" -- and dropped every diagnostic a test
        # prints next to a failure. A filter is a place where information goes to die; this one
        # ate the only useful output of an MSVC-only failure and cost a CI round trip.
        grep -E "^  (FAIL|broken|XPASS|info)" "$TMP/$t.out" | sed 's/^/  /'
    done
done

echo
[ $rc -eq 0 ] && echo "all feature levels: ok" || echo "SOMETHING FAILED"
exit $rc
