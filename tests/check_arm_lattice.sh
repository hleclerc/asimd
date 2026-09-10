#!/usr/bin/env bash
# IS THE ARM FEATURE LATTICE WHERE IT CLAIMS TO BE?
#
# `ArmCpuFeatures.h` splits Advanced SIMD in two: `NEON`, everything an ARMv7-A part has, and
# `ASIMD`, what only AArch64 adds. Every registration in the backend is guarded by one or the
# other, and getting that boundary wrong is the failure mode the x86 lattice was rewritten for:
# an intrinsic guarded too loosely compiles fine on the machine it was written on and not at all
# on the target that needs it.
#
# So the boundary gets checked from BOTH sides:
#
#   arm_lattice_neon.c    everything under `NEON`  -- must compile for ARMv7 *and* AArch64
#   arm_lattice_asimd.c   everything under `ASIMD` -- must compile for AArch64 and NOT for ARMv7
#
# The second is not decoration. If those intrinsics were available on ARMv7 too, the `ASIMD`
# guard would be denying a 32-bit part instructions it has; and if a later edit moved one of them
# under `NEON`, the first file would start failing on ARMv7 and name it.
#
# NO CROSS TOOLCHAIN AND NO SYSROOT NEEDED. `arm_neon.h` comes from the compiler, and both files
# are C with no standard library, so `clang` alone answers the question on any host -- which is
# what makes this runnable on a laptop rather than only in CI.
#
#   ./check_arm_lattice.sh              clang
#   ./check_arm_lattice.sh clang-18
#
set -u
CC=${1:-clang}
HERE=$( cd "$( dirname "$0" )" && pwd )
TMP=$( mktemp -d ); trap 'rm -rf "$TMP"' EXIT

command -v "$CC" >/dev/null || { echo "$CC not found"; exit 1; }

V7=( --target=armv7a-linux-gnueabihf -mfpu=neon -mfloat-abi=hard )
A64=( --target=aarch64-linux-gnu )

try_build() { # $1 file  $2... flags
    local f=$1; shift
    "$CC" -O1 -c "$HERE/$f" -o "$TMP/out.o" "$@" 2>"$TMP/err.txt"
}

rc=0
report() { # $1 label  $2 expected(yes|no)  $3 actual(0|1 from try_build)
    local got; [ "$3" = 0 ] && got=yes || got=no
    if [ "$got" = "$2" ]; then printf '  %-46s %-3s  ok\n' "$1" "$got"
    else printf '  %-46s %-3s  EXPECTED %s\n' "$1" "$got" "$2"; rc=1
         sed 's/^/      /' "$TMP/err.txt" | grep -m3 error: || true; fi
}

echo "$CC: does the ARM feature lattice hold?"

try_build arm_lattice_neon.c  "${V7[@]}";  report "NEON-guarded intrinsics on ARMv7-A"  yes $?
try_build arm_lattice_neon.c  "${A64[@]}"; report "NEON-guarded intrinsics on AArch64"   yes $?
try_build arm_lattice_asimd.c "${A64[@]}"; report "ASIMD-guarded intrinsics on AArch64"  yes $?
try_build arm_lattice_asimd.c "${V7[@]}";  report "ASIMD-guarded intrinsics on ARMv7-A"  no  $?

echo
[ $rc -eq 0 ] && echo "the lattice is where ArmCpuFeatures.h says it is" \
              || echo "THE LATTICE IS WRONG: a feature guard does not match the instruction set"
exit $rc
