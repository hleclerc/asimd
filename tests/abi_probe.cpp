// THE ABI PROBE. It asserts nothing itself -- it is READ at the disassembler, because the
// regression it catches shows up nowhere else: the code stays correct, it merely becomes twice as
// slow because every value goes through the stack. See § 3 of the README.
//
// `xmake.lua` disassembles each `probe_*` below and counts the instructions that touch the stack.
//
// TWO GROUPS, AND ONLY THE FIRST ONE IS ENFORCEABLE. That distinction was missing, and it made
// this check fail CI for a correct reason nobody had stated:
//
//   THE ABI QUESTION ONLY HAS A RIGHT ANSWER AT A WIDTH THAT IS ONE REGISTER. Above that, NEITHER
//   convention can pass the value in registers, and it is not a layout bug -- it is the ABI:
//
//     SysV x86-64   an aggregate travels in registers only if its first eightbyte is SSE and all
//                   the following ones are SSEUP. Two `__m256d` give SSE,SSEUP,SSEUP,SSEUP then
//                   SSE again -- the fifth eightbyte alone sends the whole thing to memory. The
//                   README says exactly this about `Split`: "unrecoverable".
//     AAPCS64       a Homogeneous Vector Aggregate is one to FOUR members of the same vector
//                   type. Beyond that the argument is passed BY REFERENCE, in `x0`-`x7`, and the
//                   result returned indirectly through `x8`.
//
//   So `SimdVec<double,8>` -- 64 bytes -- crosses a call through memory on every target without a
//   64-byte register. It was clean on the one machine this probe was written on, a Skylake-X,
//   where 8 doubles ARE one `zmm`; on a CI runner with AVX2 it is 18 instructions and 12 stack
//   accesses, and the check failed the build for something no code change can fix.
//
// HENCE THE FIRST GROUP IS WRITTEN AT THE NATIVE WIDTH, `SimdVec<float>` and friends, with no
// number in the type. `SimdSize<T>` is by definition one register: four lanes under SSE2 and on
// every ARM part, eight under AVX2, sixteen under AVX-512. That is precisely the set where "in a
// register or through memory" is a question with a right answer, on every target, by construction
// rather than by coincidence -- and it is the width the README's own argument is about.
//
// TWO OF THESE DID COME OUT DIRTY when the probe was first widened, and the two causes were
// different -- which is why it is worth several symbols rather than one:
//   `probe_fma_f64`   23 instructions, 12 stack accesses. Not an ABI problem at all: no `fma`
//                     was registered at eight lanes of double, so the generic lane loop ran.
//   `probe_sel_lane`   9 instructions,  3 stack accesses. A real ABI problem: the lane-mask impl
//                     still held an array and a Split in its union.
// The first of those is now caught where it belongs, by `require_at_least` in the dispatch tests
// -- a static_assert, which names the operation instead of leaving a number to be interpreted.
#include <asimd/asimd.h>

// =============================================================================================
// GROUP ONE -- EXACTLY ONE REGISTER, on whatever this target is. ENFORCED: any of these touching
// the stack fails the build.
// =============================================================================================
using VN = asimd::SimdVec<float>;                       // SimdSize<float> lanes: one register
using DN = asimd::SimdVec<double>;                      // idem
using IN = asimd::SimdVec<asimd::SI32>;                 // idem

extern "C" VN probe_nat_fma    ( VN a, VN b, VN c ) { return asimd::fma( a, b, c ); }
extern "C" DN probe_nat_fma_f64( DN a, DN b, DN c ) { return asimd::fma( a, b, c ); }
extern "C" VN probe_nat_perm   ( VN a, IN i )       { return asimd::permute( a, i ); }
extern "C" IN probe_nat_add    ( IN a, IN b )       { return a + b; }

// ---- and so does a mask -- IN THE FLAVOUR THIS TARGET ACTUALLY PRODUCES ---------------------
//
// The lane flavour is the one that regressed: `PI32 values[ 8 ]` classifies SSE,SSE,SSE,SSE, a
// union takes the worst class among its members, and the whole mask went to memory.
//
// DEDUCED FROM `gt`, NOT WRITTEN OUT, and for the same reason the widths above are `SimdSize<T>`
// rather than a number. Spelling it `SimdMask<NF,32>` asks for a LANE mask at the native width,
// and on AVX-512 that is sixteen 32-bit lanes -- 64 bytes, a flavour that target never produces
// (its comparisons return `k` registers) and for which there is therefore no register impl. The
// probe then failed on a configuration nothing in the library would ever construct.
//
// `decltype( gt( ... ) )` is whichever flavour the selector picked: a `k` register on AVX-512, a
// `__m256i` on AVX2, a `__m128i` on SSE2, a `uint32x4_t` on ARM. All four are one register, and
// all four are what `select` will really be handed.
using MN = decltype( asimd::gt( VN(), VN() ) );
using MD = decltype( asimd::gt( DN(), DN() ) );

extern "C" VN probe_nat_sel    ( MN m, VN a, VN b ) { return asimd::select( m, a, b ); }
extern "C" DN probe_nat_sel_f64( MD m, DN a, DN b ) { return asimd::select( m, a, b ); }

// =============================================================================================
// GROUP TWO -- A FIXED width of eight lanes, which is one register on AVX2 and up and two or more
// everywhere else, ARM included. REPORTED, NOT ENFORCED, for the reason at the top of this file.
//
// They are still worth disassembling. What they show is not the ABI but whether the SPLIT path
// keeps register-level operations underneath it: two `vfmadd213pd` in `probe_fma_f64` above, four
// `fmla.2d` on ARM, and a lane loop if a registration is ever lost. That last case is now also a
// `require_at_least` in both dispatch tests, which is where a lost registration should be caught
// -- it fails the build with the operation's name rather than with a number to be squinted at.
// =============================================================================================
using V8 = asimd::SimdVec<float,8>;
using D8 = asimd::SimdVec<double,8>;
using I8 = asimd::SimdVec<asimd::SI32,8>;

extern "C" V8 probe_fma    ( V8 a, V8 b, V8 c ) { return asimd::fma( a, b, c ); }
extern "C" D8 probe_fma_f64( D8 a, D8 b, D8 c ) { return asimd::fma( a, b, c ); }
extern "C" V8 probe_perm   ( V8 a, I8 i )       { return asimd::permute( a, i ); }
extern "C" V8 probe_sel_lane( asimd::SimdMask<8,32> m, V8 a, V8 b ) { return asimd::select( m, a, b ); }
extern "C" V8 probe_sel_bits( asimd::SimdMask<8, 1> m, V8 a, V8 b ) { return asimd::select( m, a, b ); }
