// THE ABI PROBE. It asserts nothing -- it is READ at the disassembler, because the regression it
// catches shows up nowhere else: the code stays correct, it merely becomes twice as slow because
// every value goes through the stack. See § 3 of the README.
//
// `xmake.lua` disassembles each `probe_*` below and counts the instructions that touch %rsp. All
// of them must come out at zero; any one that does not fails the build.
//
// TWO OF THESE DID NOT, when the probe was widened from one symbol to five, and the two causes
// were different -- which is why the probe is worth five symbols rather than one:
//   `probe_fma_f64`   23 instructions, 12 stack accesses. Not an ABI problem at all: no `fma`
//                     was registered at eight lanes of double, so the generic lane loop ran.
//   `probe_sel_lane`   9 instructions,  3 stack accesses. A real ABI problem: the lane-mask impl
//                     still held an array and a Split in its union.
#include <asimd/SimdOpsPlus.h>

using V = asimd::SimdVec<float,8>;
using D = asimd::SimdVec<double,8>;
using I = asimd::SimdVec<asimd::SI32,8>;

// ---- a vector crosses a call in its registers -----------------------------------------------
extern "C" V probe_fma    ( V a, V b, V c ) { return asimd::fma( a, b, c ); }
extern "C" D probe_fma_f64( D a, D b, D c ) { return asimd::fma( a, b, c ); }
extern "C" V probe_perm   ( V a, I i )      { return asimd::permute( a, i ); }

// ---- and so does a mask, in both flavours ---------------------------------------------------
// The lane flavour is the one that regressed: `PI32 values[ 8 ]` classifies SSE,SSE,SSE,SSE, a
// union takes the worst class among its members, and the whole mask went to memory. The
// bit-flavoured (AVX-512) mask holds no array and was already fine.
extern "C" V probe_sel_lane( asimd::SimdMask<8,32> m, V a, V b ) { return asimd::select( m, a, b ); }
extern "C" V probe_sel_bits( asimd::SimdMask<8, 1> m, V a, V b ) { return asimd::select( m, a, b ); }
