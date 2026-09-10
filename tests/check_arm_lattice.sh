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
# NO CROSS TOOLCHAIN NEEDED, and `-ffreestanding` is what makes that true off macOS. `arm_neon.h`
# comes from the compiler, but it includes `<stdint.h>`, and clang's own `stdint.h` defers to the
# system one whenever `__STDC_HOSTED__` is set. On an aarch64 Linux host that lands in the HOST's
# glibc and dies on `bits/libc-header-start.h`, because the armhf multiarch headers are not
# installed. `-ffreestanding` clears `__STDC_HOSTED__` and clang defines the integer types itself.
#
# AND IF IT STILL CANNOT COMPILE, THIS SCRIPT SAYS SO INSTEAD OF BLAMING THE LATTICE. That
# distinction is the whole reason for the `cannot run` verdict below: the first version reported
#
#     NEON-guarded intrinsics on ARMv7-A             no   EXPECTED yes
#     THE LATTICE IS WRONG: a feature guard does not match the instruction set
#
# when the truth was a missing header. A check that reports the wrong conclusion is worse than one
# that reports none -- someone goes and looks for a bug in `ArmCpuFeatures.h` that is not there.
# The two are told apart by whether a FILE could not be found. Not by `fatal error:` alone: the
# ASIMD-on-ARMv7 row is *expected* to fail, it fails with dozens of errors, and clang caps that
# with `fatal error: too many errors emitted` -- which the first attempt at this discriminator
# duly misread as "cannot run". Hence also `-ferror-limit=0`, so that cap never fires.
#
#   ./check_arm_lattice.sh                       clang
#   ./check_arm_lattice.sh clang-18
#   ./check_arm_lattice.sh clang --strict         a SKIPPED row is a failure too
#
# `--strict` is what CI uses. A skip is the honest verdict for a host that genuinely cannot
# compile for the target, and it must not fail a contributor's build -- but in CI we KNOW the
# check can run, so a skip there means something changed underneath it, and a green tick nobody
# reads is how the ARMv7 row would quietly stop being checked at all.
set -u
CC=${1:-clang}
STRICT=${2:-}
HERE=$( cd "$( dirname "$0" )" && pwd )
TMP=$( mktemp -d ); trap 'rm -rf "$TMP"' EXIT

command -v "$CC" >/dev/null || { echo "$CC not found"; exit 1; }

V7=( --target=armv7a-linux-gnueabihf -mfpu=neon -mfloat-abi=hard )
A64=( --target=aarch64-linux-gnu )

# `-ffreestanding`: see the note above. `-Wno-incompatible-pointer-types` is not needed and not
# passed -- the probe files use the `<stdint.h>` typedefs precisely so the signatures match.
try_build() { # $1 file  $2... flags -- prints "yes" / "no" / "cannot run"
    local f=$1; shift
    if "$CC" -O1 -ffreestanding -ferror-limit=0 -c "$HERE/$f" -o "$TMP/out.o" "$@" 2>"$TMP/err.txt"; then
        echo yes
    elif grep -qE "file not found|No such file or directory|unknown target|invalid target|unsupported option" "$TMP/err.txt"; then
        echo "cannot run"
    else
        echo no
    fi
}

rc=0
skipped=0
report() { # $1 label  $2 expected  $3 actual
    if [ "$3" = "cannot run" ]; then
        printf '  %-46s %-10s  SKIPPED -- this host cannot compile it\n' "$1" "$3"
        sed 's/^/      /' "$TMP/err.txt" | grep -m2 -E "file not found|No such file|unknown target|invalid target|unsupported option" || true
        skipped=1
    elif [ "$3" = "$2" ]; then
        printf '  %-46s %-10s  ok\n' "$1" "$3"
    else
        printf '  %-46s %-10s  EXPECTED %s\n' "$1" "$3" "$2"
        sed 's/^/      /' "$TMP/err.txt" | grep -m3 "error:" || true
        rc=1
    fi
}

echo "$CC: does the ARM feature lattice hold?"

report "NEON-guarded intrinsics on ARMv7-A"  yes "$( try_build arm_lattice_neon.c  "${V7[@]}"  )"
report "NEON-guarded intrinsics on AArch64"  yes "$( try_build arm_lattice_neon.c  "${A64[@]}" )"
report "ASIMD-guarded intrinsics on AArch64" yes "$( try_build arm_lattice_asimd.c "${A64[@]}" )"
report "ASIMD-guarded intrinsics on ARMv7-A" no  "$( try_build arm_lattice_asimd.c "${V7[@]}" )"

echo
if [ $rc -ne 0 ]; then
    echo "THE LATTICE IS WRONG: a feature guard does not match the instruction set"
elif [ $skipped -ne 0 ]; then
    # NOT an error, and NOT silence either. The rows that did run are still checked above; what is
    # reported here is that one of them could not, and which.
    echo "the lattice holds where it could be checked -- one or more rows were SKIPPED above"
    [ "$STRICT" = "--strict" ] && { echo "(--strict: a skipped row counts as a failure)"; rc=1; }
else
    echo "the lattice is where ArmCpuFeatures.h says it is"
fi
exit $rc
