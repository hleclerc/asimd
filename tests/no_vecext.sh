#!/usr/bin/env bash
# WHAT THE DISPATCH TABLE IS ACTUALLY WORTH.
#
# gcc and clang have `vector_size`; MSVC does not. Where the table has no entry, the generic
# fallback applies the operation to the whole `values` vector and gcc lowers it to the right
# instruction on its own -- which is free, and which also HIDES every hole in the table. MSVC gets
# a scalar lane loop instead.
#
# So: compile each cell twice, once normally and once with ASIMD_NO_COMPILER_VECTORS (which is
# exactly MSVC's situation), and count instructions. A cell that is the same either way is carried
# by the table. A cell that blows up is carried by gcc, and is a hole.
#
#   ./no_vecext.sh                 -march=native
#   ./no_vecext.sh "-msse2"
#
set -u
FLAGS=${1:--march=native}
CXX=${2:-g++}
HERE=$( cd "$( dirname "$0" )" && pwd )
TMP=$( mktemp -d ); trap 'rm -rf "$TMP"' EXIT

# name|declaration|body   -- `a`, `b`, `c` are SimdVec<T,N>, `i` is SimdVec<SI32,N>
OPS=(
  "add|V|return a + b;"
  "sub|V|return a - b;"
  "mul|V|return a * b;"
  "div|V|return a / b;"
  "and|V|return a & b;"
  "min|V|return min( a, b );"
  "max|V|return max( a, b );"
  "sum|T|return a.sum();"
  "fma|V|return fma( a, b, c );"
  "tobits|PI64|return to_bits( gt( a, b ) );"
  "select|V|return select( gt( a, b ), a, c );"
  "permute|V|return permute( a, i );"
  "bcast|V|return bcast_lane<1>( a );"
  "iota|V|return V::iota( T( 3 ) );"
)
CELLS=( "FP32 4" "FP32 8" "FP32 16" "FP64 2" "FP64 4" "FP64 8"
        "SI32 4" "SI32 8" "SI32 16" "PI32 8" "SI64 4" "SI64 8" )

count() { objdump -d --disassemble=probe "$1" 2>/dev/null | grep -cE '^\s+[0-9a-f]+:'; }

# EVERYTHING GOES THROUGH POINTERS, on purpose. Taking the arguments by value would measure the
# wrong thing: without `vector_size` the impl holds an array, an array classifies MEMORY on the
# SysV ABI, and every function grows four loads and a store -- swamping the difference we are
# after. (On MSVC that is moot: its x64 convention passes these by pointer either way.) With
# pointers in and a pointer out, both builds do the same loads and the same store, and what is
# left is the computation.
build() { # $1 out  $2 extra-defines  $3 T  $4 N  $5 ret  $6 body
  cat > "$1.cpp" <<EOF
#include <asimd/SimdOpsPlus.h>
using namespace asimd;
using T = $3; static constexpr int NL = $4;
using V = SimdVec<T,NL>;
extern "C" void probe( const V *pa, const V *pb, const V *pc, const SimdVec<SI32,NL> *pi, $5 *out ) {
    const V a = *pa, b = *pb, c = *pc; const SimdVec<SI32,NL> i = *pi;
    (void)b; (void)c; (void)i;
    *out = [ & ] () -> $5 { $6 }();
}
EOF
  $CXX -std=c++20 -O2 $FLAGS $2 -I"$HERE/../src" -c "$1.cpp" -o "$1.o" 2>/dev/null
}

printf '%s, %s\n\n' "$CXX $FLAGS" "instructions: with compiler vectors -> without (the MSVC path)"
printf '%-9s' ""; for c in "${CELLS[@]}"; do printf '%-11s' "${c// /x}"; done; printf '\n'

worse=0; total=0
for op in "${OPS[@]}"; do
    IFS='|' read -r name ret body <<< "$op"
    printf '%-9s' "$name"
    for cell in "${CELLS[@]}"; do
        read -r ty n <<< "$cell"
        r=$( [ "$ret" = V ] && echo V || echo "$ret" )
        if build "$TMP/w"                                "" "$ty" "$n" "$r" "$body" &&
           build "$TMP/n" "-DASIMD_NO_COMPILER_VECTORS"     "$ty" "$n" "$r" "$body"; then
            a=$( count "$TMP/w.o" ); b=$( count "$TMP/n.o" )
            total=$(( total + 1 ))
            if [ "$b" -gt $(( a + a / 4 + 1 )) ]; then worse=$(( worse + 1 )); printf '%-11s' "$a->$b !!"
            else printf '%-11s' "$a->$b"; fi
        else printf '%-11s' "  n/a"; fi
    done; printf '\n'
done

printf '\n%d of %d cells degrade without the compiler vectors (marked !!).\n' "$worse" "$total"
cat <<'NOTE'

Integer `div` is expected to stay marked, at any width: x86 has NO SIMD integer division. Both
paths end up scalarising; with a vector type gcc reaches for a reciprocal-multiply sequence it
cannot apply to an array. Nothing in the dispatch table can fix that, because there is no
instruction to dispatch to. Anything else marked here is a hole in the table.
NOTE
