#pragma once

// =============================================================================================
// REGISTER-LEVEL VARIANTS FOR x86.
//
// Each one replaces the generic form -- which stays correct, and stays the only one available
// elsewhere -- with one or two instructions. Ranks do the sorting, so an AVX-512 form and an AVX
// one coexist at the same width without the ambiguity plain overloading would produce.
//
// WHAT THIS FILE USED TO COVER: `float x 8`. Nothing else. Not `float x 4`, not `float x 16`, no
// `double` at any width, no integer type except `SI32 x 8` for `eq`/`ge`/`permute`. 55 of the 63
// cells of the grid `tests/test_x86_dispatch.cpp` prints were on the scalar path -- INCLUDING
// THE NATIVE WIDTH on an AVX-512 machine, which is 16 floats. So `SimdVec<float>`, written with
// no explicit width, got the slowest path on the widest hardware: `fma` measured 23 instructions
// with 12 stack accesses against 3 and 0 at eight lanes.
//
// The mechanism was never the problem -- `require_at_least` would have caught every one of them,
// and `test_x86_dispatch.cpp` now calls it at each width the library claims to support. What was
// missing was the registrations. Hence this file, laid out by feature level:
//
//   1. 128 bits -- SSE2, SSE4.1, FMA
//   2. 256 bits -- AVX, AVX2, FMA
//   3. 512 bits -- AVX-512F
//   4. mask registers at 128 and 256 bits -- AVX-512VL
//
// A note on the two flavours of mask. Below AVX-512 a comparison yields a LANE mask (all-ones
// words) and `select` is a `blendv`; from AVX-512 on it yields a BIT mask in a `k` register and
// `select` is a masked move. Both are registered, at ranks REGISTER and MASK_REGISTER, and the
// caller never has to name which -- `gt(...)` returns whichever the target has, and `select` and
// `to_bits` accept both. That is what the `IS` slot of `Key` carries.
// =============================================================================================

#include "impl/x86_intrin.h"

#if defined( _M_X64 ) || defined( _M_IX86 ) || defined( __i386__ ) || defined( __x86_64__ )

#include "architectures/X86CpuFeatures.h"

// The variant SHAPES -- one macro per (operation, mask flavour) -- now live in
// `SimdOpsPlus_Shapes.h`, because `SimdOpsPlus_Neon.h` needs the identical ones: what a variant
// of `select` looks like is a property of `Selection.h`, not of the instruction set. What is left
// in this file is the TABLE, which is the part that is actually about x86.
#include "SimdOpsPlus_Shapes.h"

namespace asimd {

// =============================================================================================
// 1. 128 BITS -- SSE2, SSE4.1, FMA
// =============================================================================================
#ifdef ASIMD_X86_HAS_SSE2

// ---- comparisons, lane flavour. `cmpps` / `pcmpgtd` are SSE2; there is no predicate operand,
// so each relation is its own instruction, and `ge` on the integer types is `not( a < b )`.
ASIMD_PLUS_CMP( SSE2, cmp_gt, FP32, 4, 32, REGISTER, _mm_castps_si128( _mm_cmpgt_ps( a.data.reg, b.data.reg ) ) );
ASIMD_PLUS_CMP( SSE2, cmp_lt, FP32, 4, 32, REGISTER, _mm_castps_si128( _mm_cmplt_ps( a.data.reg, b.data.reg ) ) );
ASIMD_PLUS_CMP( SSE2, cmp_eq, FP32, 4, 32, REGISTER, _mm_castps_si128( _mm_cmpeq_ps( a.data.reg, b.data.reg ) ) );
ASIMD_PLUS_CMP( SSE2, cmp_ge, FP32, 4, 32, REGISTER, _mm_castps_si128( _mm_cmpge_ps( a.data.reg, b.data.reg ) ) );
ASIMD_PLUS_CMP( SSE2, cmp_gt, FP64, 2, 64, REGISTER, _mm_castpd_si128( _mm_cmpgt_pd( a.data.reg, b.data.reg ) ) );
ASIMD_PLUS_CMP( SSE2, cmp_lt, FP64, 2, 64, REGISTER, _mm_castpd_si128( _mm_cmplt_pd( a.data.reg, b.data.reg ) ) );
ASIMD_PLUS_CMP( SSE2, cmp_eq, FP64, 2, 64, REGISTER, _mm_castpd_si128( _mm_cmpeq_pd( a.data.reg, b.data.reg ) ) );
ASIMD_PLUS_CMP( SSE2, cmp_ge, FP64, 2, 64, REGISTER, _mm_castpd_si128( _mm_cmpge_pd( a.data.reg, b.data.reg ) ) );

ASIMD_PLUS_CMP( SSE2, cmp_eq, SI32, 4, 32, REGISTER, _mm_cmpeq_epi32( a.data.reg, b.data.reg ) );
ASIMD_PLUS_CMP( SSE2, cmp_gt, SI32, 4, 32, REGISTER, _mm_cmpgt_epi32( a.data.reg, b.data.reg ) );
ASIMD_PLUS_CMP( SSE2, cmp_lt, SI32, 4, 32, REGISTER, _mm_cmpgt_epi32( b.data.reg, a.data.reg ) );
ASIMD_PLUS_CMP( SSE2, cmp_ge, SI32, 4, 32, REGISTER, _mm_xor_si128( _mm_cmpgt_epi32( b.data.reg, a.data.reg ), _mm_set1_epi32( -1 ) ) );

// ---- to_bits. `movmskps` is SSE, `movmskpd` SSE2: the lane mask goes straight to an integer.
ASIMD_PLUS_TO_BITS( SSE2, 4, 32, REGISTER, PI64( unsigned( _mm_movemask_ps( _mm_castsi128_ps( m.data.reg ) ) ) ) );
ASIMD_PLUS_TO_BITS( SSE2, 2, 64, REGISTER, PI64( unsigned( _mm_movemask_pd( _mm_castsi128_pd( m.data.reg ) ) ) ) );

// ---- mask_from_bits, lane flavour. Broadcast the integer, keep one bit per lane, compare it
// back to itself: three instructions where the generic form writes four lanes one at a time.
ASIMD_PLUS_MASK_FROM_BITS( SSE2, 4, 32, REGISTER, ( [ & ] {
    const __m128i sel = _mm_setr_epi32( 1, 2, 4, 8 );
    return _mm_cmpeq_epi32( _mm_and_si128( _mm_set1_epi32( int( PI32( b ) ) ), sel ), sel ); }() ) );

#ifdef ASIMD_X86_HAS_SSE4_1
// ---- select. `blendv` is SSE4.1; below it the generic form applies.
ASIMD_PLUS_SELECT( SSE4_1, FP32, 4, 32, REGISTER, _mm_blendv_ps( b.data.reg, a.data.reg, _mm_castsi128_ps( m.data.reg ) ) );
ASIMD_PLUS_SELECT( SSE4_1, FP64, 2, 64, REGISTER, _mm_blendv_pd( b.data.reg, a.data.reg, _mm_castsi128_pd( m.data.reg ) ) );
ASIMD_PLUS_SELECT( SSE4_1, SI32, 4, 32, REGISTER, _mm_castps_si128( _mm_blendv_ps( _mm_castsi128_ps( b.data.reg ), _mm_castsi128_ps( a.data.reg ), _mm_castsi128_ps( m.data.reg ) ) ) );
ASIMD_PLUS_SELECT( SSE4_1, PI32, 4, 32, REGISTER, _mm_castps_si128( _mm_blendv_ps( _mm_castsi128_ps( b.data.reg ), _mm_castsi128_ps( a.data.reg ), _mm_castsi128_ps( m.data.reg ) ) ) );
ASIMD_PLUS_SELECT( SSE4_1, SI64, 2, 64, REGISTER, _mm_castpd_si128( _mm_blendv_pd( _mm_castsi128_pd( b.data.reg ), _mm_castsi128_pd( a.data.reg ), _mm_castsi128_pd( m.data.reg ) ) ) );
ASIMD_PLUS_SELECT( SSE4_1, PI64, 2, 64, REGISTER, _mm_castpd_si128( _mm_blendv_pd( _mm_castsi128_pd( b.data.reg ), _mm_castsi128_pd( a.data.reg ), _mm_castsi128_pd( m.data.reg ) ) ) );
#endif

// ---- bcast_lane. `shufps` with a repeated index; one instruction, and no memory.
ASIMD_PLUS_BCAST( SSE2, FP32, 4, _mm_shuffle_ps( v.data.reg, v.data.reg, _MM_SHUFFLE( LANE, LANE, LANE, LANE ) ) );
ASIMD_PLUS_BCAST( SSE2, SI32, 4, _mm_shuffle_epi32( v.data.reg, _MM_SHUFFLE( LANE, LANE, LANE, LANE ) ) );
ASIMD_PLUS_BCAST( SSE2, PI32, 4, _mm_shuffle_epi32( v.data.reg, _MM_SHUFFLE( LANE, LANE, LANE, LANE ) ) );
ASIMD_PLUS_BCAST( SSE2, FP64, 2, _mm_shuffle_pd   ( v.data.reg, v.data.reg, ( LANE << 1 ) | LANE ) );
ASIMD_PLUS_BCAST( SSE2, SI64, 2, _mm_shuffle_epi32( v.data.reg, _MM_SHUFFLE( 2*LANE+1, 2*LANE, 2*LANE+1, 2*LANE ) ) );
ASIMD_PLUS_BCAST( SSE2, PI64, 2, _mm_shuffle_epi32( v.data.reg, _MM_SHUFFLE( 2*LANE+1, 2*LANE, 2*LANE+1, 2*LANE ) ) );

#ifdef ASIMD_X86_HAS_SSSE3
// ---- permute at 128 bits WITHOUT AVX. `pshufb` shuffles bytes, so a 32-bit lane index has to be
// turned into four consecutive byte indices: multiply by four, broadcast the low byte of each
// lane across its four bytes, then add 0,1,2,3. Four instructions instead of one `vpermilps`,
// and it is what makes the split form below reachable on a plain SSE machine -- without it,
// `SimdVec<float,8>` on SSE2 had no permutation better than a round trip through memory.
#define ASIMD_PLUS_PSHUFB_32( T ) \
    ASIMD_PLUS_PERMUTE_EXCL( SSSE3, AVX, T, 4, ( [ & ] { \
        const __m128i bcast = _mm_setr_epi8( 0,0,0,0, 4,4,4,4, 8,8,8,8, 12,12,12,12 ); \
        const __m128i lane  = _mm_setr_epi8( 0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3 ); \
        __m128i c = _mm_shuffle_epi8( _mm_slli_epi32( idx.data.reg, 2 ), bcast ); \
        c = _mm_add_epi8( c, lane ); \
        return ASIMD_PSHUFB_AS( T )( c ); }() ) )

#define ASIMD_PSHUFB_AS_FP32( c ) _mm_castsi128_ps( _mm_shuffle_epi8( _mm_castps_si128( v.data.reg ), c ) )
#define ASIMD_PSHUFB_AS_SI32( c ) _mm_shuffle_epi8( v.data.reg, c )
#define ASIMD_PSHUFB_AS_PI32( c ) _mm_shuffle_epi8( v.data.reg, c )
#define ASIMD_PSHUFB_AS( T ) ASIMD_PSHUFB_AS_##T

ASIMD_PLUS_PSHUFB_32( FP32 );
ASIMD_PLUS_PSHUFB_32( SI32 );
ASIMD_PLUS_PSHUFB_32( PI32 );

#undef ASIMD_PSHUFB_AS
#undef ASIMD_PSHUFB_AS_PI32
#undef ASIMD_PSHUFB_AS_SI32
#undef ASIMD_PSHUFB_AS_FP32
#undef ASIMD_PLUS_PSHUFB_32
#endif // ASIMD_X86_HAS_SSSE3

#ifdef ASIMD_X86_HAS_AVX
// ---- permute at 128 bits. `vpermilps` takes its control from a register, which is what makes it
// a VARIABLE-index permutation, and it beats the `pshufb` sequence above by three instructions.
ASIMD_PLUS_PERMUTE( AVX, FP32, 4, _mm_permutevar_ps( v.data.reg, idx.data.reg ) );
ASIMD_PLUS_PERMUTE( AVX, SI32, 4, _mm_castps_si128( _mm_permutevar_ps( _mm_castsi128_ps( v.data.reg ), idx.data.reg ) ) );
ASIMD_PLUS_PERMUTE( AVX, PI32, 4, _mm_castps_si128( _mm_permutevar_ps( _mm_castsi128_ps( v.data.reg ), idx.data.reg ) ) );
#endif

#ifdef ASIMD_X86_HAS_FMA
ASIMD_PLUS_FMA( SSE2, FMA, FP32, 4, _mm_fmadd_ps );
ASIMD_PLUS_FMA( SSE2, FMA, FP64, 2, _mm_fmadd_pd );
#endif

#endif // ASIMD_X86_HAS_SSE2

// =============================================================================================
// 2. 256 BITS -- AVX, AVX2, FMA
// =============================================================================================
#ifdef ASIMD_X86_HAS_AVX

// ---- comparisons. AVX finally has a predicate operand, but only for floating point: the 256-bit
// integer compares are AVX2, and stay signed-only there.
#define ASIMD_PLUS_AVX_FP_CMP( TAG, PRED ) \
    ASIMD_PLUS_CMP( AVX, TAG, FP32, 8, 32, REGISTER, _mm256_castps_si256( _mm256_cmp_ps( a.data.reg, b.data.reg, PRED ) ) ); \
    ASIMD_PLUS_CMP( AVX, TAG, FP64, 4, 64, REGISTER, _mm256_castpd_si256( _mm256_cmp_pd( a.data.reg, b.data.reg, PRED ) ) )

ASIMD_PLUS_AVX_FP_CMP( cmp_gt, _CMP_GT_OQ );
ASIMD_PLUS_AVX_FP_CMP( cmp_lt, _CMP_LT_OQ );
ASIMD_PLUS_AVX_FP_CMP( cmp_eq, _CMP_EQ_OQ );
ASIMD_PLUS_AVX_FP_CMP( cmp_ge, _CMP_GE_OQ );

#undef ASIMD_PLUS_AVX_FP_CMP

// ---- to_bits and select at 256 bits. `vblendvps` is AVX -- unlike its 128-bit ancestor, which
// needs SSE4.1 -- so no extra guard here.
ASIMD_PLUS_TO_BITS( AVX, 8, 32, REGISTER, PI64( unsigned( _mm256_movemask_ps( _mm256_castsi256_ps( m.data.reg ) ) ) ) );
ASIMD_PLUS_TO_BITS( AVX, 4, 64, REGISTER, PI64( unsigned( _mm256_movemask_pd( _mm256_castsi256_pd( m.data.reg ) ) ) ) );

ASIMD_PLUS_SELECT( AVX, FP32, 8, 32, REGISTER, _mm256_blendv_ps( b.data.reg, a.data.reg, _mm256_castsi256_ps( m.data.reg ) ) );
ASIMD_PLUS_SELECT( AVX, FP64, 4, 64, REGISTER, _mm256_blendv_pd( b.data.reg, a.data.reg, _mm256_castsi256_pd( m.data.reg ) ) );

#define ASIMD_PLUS_AVX_ISELECT( T, N, IS, TO, BACK, BLEND ) \
    ASIMD_PLUS_SELECT( AVX, T, N, IS, REGISTER, BACK( BLEND( TO( b.data.reg ), TO( a.data.reg ), TO( m.data.reg ) ) ) )
ASIMD_PLUS_AVX_ISELECT( SI32, 8, 32, _mm256_castsi256_ps, _mm256_castps_si256, _mm256_blendv_ps );
ASIMD_PLUS_AVX_ISELECT( PI32, 8, 32, _mm256_castsi256_ps, _mm256_castps_si256, _mm256_blendv_ps );
ASIMD_PLUS_AVX_ISELECT( SI64, 4, 64, _mm256_castsi256_pd, _mm256_castpd_si256, _mm256_blendv_pd );
ASIMD_PLUS_AVX_ISELECT( PI64, 4, 64, _mm256_castsi256_pd, _mm256_castpd_si256, _mm256_blendv_pd );
#undef ASIMD_PLUS_AVX_ISELECT

#ifdef ASIMD_X86_HAS_FMA
ASIMD_PLUS_FMA( AVX, FMA, FP32, 8, _mm256_fmadd_ps );
ASIMD_PLUS_FMA( AVX, FMA, FP64, 4, _mm256_fmadd_pd );
#endif

// ---- a FULL 8-lane permutation on AVX, without AVX2 ------------------------------------------
//
// `vpermps` is AVX2. AVX has only `vpermilps`, which permutes inside each 128-bit half. So:
// permute in place, permute again with the halves swapped, and blend by asking whether each
// index points at the half it started in. That question is `( idx ^ lane_half ) & 4`, and the
// shift that moves bit 2 to the sign bit is done per 128-bit half because `vpslld` at 256 bits
// is, once more, AVX2.
//
// Eight instructions against one, and against the eighty-one of the generic form -- which stores
// the vector to the stack, indexes it there and reloads it.
#define ASIMD_PLUS_AVX_PERM8( T, TO_PS, FROM_PS ) \
    ASIMD_PLUS_PERMUTE_EXCL( AVX, AVX2, T, 8, ( [ & ] { \
        const __m256 s = TO_PS( v.data.reg ); \
        const __m256 in_place = _mm256_permutevar_ps( s, idx.data.reg ); \
        const __m256 crossed  = _mm256_permutevar_ps( _mm256_permute2f128_ps( s, s, 0x01 ), idx.data.reg ); \
        const __m128i j0 = _mm_slli_epi32( _mm256_extractf128_si256( idx.data.reg, 0 ), 29 ); \
        const __m128i j1 = _mm_slli_epi32( _mm_xor_si128( _mm256_extractf128_si256( idx.data.reg, 1 ), \
                                                          _mm_set1_epi32( 4 ) ), 29 ); \
        const __m256 take_crossed = _mm256_castsi256_ps( \
            _mm256_insertf128_si256( _mm256_castsi128_si256( j0 ), j1, 1 ) ); \
        return FROM_PS( _mm256_blendv_ps( in_place, crossed, take_crossed ) ); }() ) )

#define ASIMD_ID_PS( x ) ( x )
ASIMD_PLUS_AVX_PERM8( FP32, ASIMD_ID_PS, ASIMD_ID_PS );
ASIMD_PLUS_AVX_PERM8( SI32, _mm256_castsi256_ps, _mm256_castps_si256 );
ASIMD_PLUS_AVX_PERM8( PI32, _mm256_castsi256_ps, _mm256_castps_si256 );
#undef ASIMD_ID_PS
#undef ASIMD_PLUS_AVX_PERM8

#ifdef ASIMD_X86_HAS_AVX2
// ---- 256-bit integer comparisons: signed in hardware, unsigned by flipping both sign bits.
ASIMD_PLUS_CMP( AVX2, cmp_eq, SI32, 8, 32, REGISTER, _mm256_cmpeq_epi32( a.data.reg, b.data.reg ) );
ASIMD_PLUS_CMP( AVX2, cmp_eq, PI32, 8, 32, REGISTER, _mm256_cmpeq_epi32( a.data.reg, b.data.reg ) );
ASIMD_PLUS_CMP( AVX2, cmp_gt, SI32, 8, 32, REGISTER, _mm256_cmpgt_epi32( a.data.reg, b.data.reg ) );
ASIMD_PLUS_CMP( AVX2, cmp_lt, SI32, 8, 32, REGISTER, _mm256_cmpgt_epi32( b.data.reg, a.data.reg ) );
ASIMD_PLUS_CMP( AVX2, cmp_ge, SI32, 8, 32, REGISTER, _mm256_xor_si256( _mm256_cmpgt_epi32( b.data.reg, a.data.reg ), _mm256_set1_epi32( -1 ) ) );
ASIMD_PLUS_CMP( AVX2, cmp_eq, SI64, 4, 64, REGISTER, _mm256_cmpeq_epi64( a.data.reg, b.data.reg ) );
ASIMD_PLUS_CMP( AVX2, cmp_eq, PI64, 4, 64, REGISTER, _mm256_cmpeq_epi64( a.data.reg, b.data.reg ) );
ASIMD_PLUS_CMP( AVX2, cmp_gt, SI64, 4, 64, REGISTER, _mm256_cmpgt_epi64( a.data.reg, b.data.reg ) );
ASIMD_PLUS_CMP( AVX2, cmp_lt, SI64, 4, 64, REGISTER, _mm256_cmpgt_epi64( b.data.reg, a.data.reg ) );
ASIMD_PLUS_CMP( AVX2, cmp_ge, SI64, 4, 64, REGISTER, _mm256_xor_si256( _mm256_cmpgt_epi64( b.data.reg, a.data.reg ), _mm256_set1_epi32( -1 ) ) );

#define ASIMD_PLUS_AVX2_UCMP( TAG, T, N, IS, W, SIGN, A, B ) \
    ASIMD_PLUS_CMP( AVX2, TAG, T, N, IS, REGISTER, _mm256_cmpgt_epi##W( \
        _mm256_xor_si256( A.data.reg, SIGN ), _mm256_xor_si256( B.data.reg, SIGN ) ) )
ASIMD_PLUS_AVX2_UCMP( cmp_gt, PI32, 8, 32, 32, _mm256_set1_epi32( int( 0x80000000u ) ), a, b );
ASIMD_PLUS_AVX2_UCMP( cmp_lt, PI32, 8, 32, 32, _mm256_set1_epi32( int( 0x80000000u ) ), b, a );
ASIMD_PLUS_AVX2_UCMP( cmp_gt, PI64, 4, 64, 64, _mm256_set1_epi64x( SI64( 0x8000000000000000ull ) ), a, b );
ASIMD_PLUS_AVX2_UCMP( cmp_lt, PI64, 4, 64, 64, _mm256_set1_epi64x( SI64( 0x8000000000000000ull ) ), b, a );
#undef ASIMD_PLUS_AVX2_UCMP

// ---- permutation across the WHOLE 256-bit register. `vpermilps` only moves lanes within each
// 128-bit half; `vpermps` crosses, which is what a permutation has to be able to do.
ASIMD_PLUS_PERMUTE( AVX2, FP32, 8, _mm256_permutevar8x32_ps( v.data.reg, idx.data.reg ) );
ASIMD_PLUS_PERMUTE( AVX2, SI32, 8, _mm256_permutevar8x32_epi32( v.data.reg, idx.data.reg ) );
ASIMD_PLUS_PERMUTE( AVX2, PI32, 8, _mm256_permutevar8x32_epi32( v.data.reg, idx.data.reg ) );

// Broadcasting lane 0 has better than a full permutation: one uop against three.
ASIMD_PLUS_BCAST( AVX2, FP32, 8, LANE == 0 ? _mm256_broadcastss_ps( _mm256_castps256_ps128( v.data.reg ) )
                                           : _mm256_permutevar8x32_ps( v.data.reg, _mm256_set1_epi32( LANE ) ) );
ASIMD_PLUS_BCAST( AVX2, SI32, 8, LANE == 0 ? _mm256_broadcastd_epi32( _mm256_castsi256_si128( v.data.reg ) )
                                           : _mm256_permutevar8x32_epi32( v.data.reg, _mm256_set1_epi32( LANE ) ) );
ASIMD_PLUS_BCAST( AVX2, PI32, 8, LANE == 0 ? _mm256_broadcastd_epi32( _mm256_castsi256_si128( v.data.reg ) )
                                           : _mm256_permutevar8x32_epi32( v.data.reg, _mm256_set1_epi32( LANE ) ) );
ASIMD_PLUS_BCAST( AVX2, FP64, 4, _mm256_permute4x64_pd   ( v.data.reg, _MM_SHUFFLE( LANE, LANE, LANE, LANE ) ) );
ASIMD_PLUS_BCAST( AVX2, SI64, 4, _mm256_permute4x64_epi64( v.data.reg, _MM_SHUFFLE( LANE, LANE, LANE, LANE ) ) );
ASIMD_PLUS_BCAST( AVX2, PI64, 4, _mm256_permute4x64_epi64( v.data.reg, _MM_SHUFFLE( LANE, LANE, LANE, LANE ) ) );

// ---- mask_from_bits, 8 lanes, lane flavour.
ASIMD_PLUS_MASK_FROM_BITS( AVX2, 8, 32, REGISTER, ( [ & ] { \
    const __m256i sel = _mm256_setr_epi32( 1, 2, 4, 8, 16, 32, 64, 128 ); \
    return _mm256_cmpeq_epi32( _mm256_and_si256( _mm256_set1_epi32( int( PI32( b ) ) ), sel ), sel ); }() ) );
#endif // ASIMD_X86_HAS_AVX2

#endif // ASIMD_X86_HAS_AVX

// =============================================================================================
// 3. 512 BITS AND MASK REGISTERS -- AVX-512F, and AVX-512VL for the 128/256-bit encodings
//
// THIS IS WHERE THE MISSING COVERAGE HURT MOST. On an AVX-512 machine `SimdSize<float>` is 16,
// so `SimdVec<float>` -- the type you get by not naming a width -- landed on a width where this
// file registered nothing at all. Every comparison returned a lane mask built one lane at a time
// and every `select` was a scalar loop.
// =============================================================================================
#ifdef ASIMD_X86_HAS_AVX512F

// ---- comparisons -> mask registers, 512 bits.
#define ASIMD_PLUS_AVX512_CMP( TAG, PRED_F, PRED_I ) \
    ASIMD_PLUS_CMP( AVX512, TAG, FP32, 16, 1, MASK_REGISTER, _mm512_cmp_ps_mask   ( a.data.reg, b.data.reg, PRED_F ) ); \
    ASIMD_PLUS_CMP( AVX512, TAG, FP64,  8, 1, MASK_REGISTER, _mm512_cmp_pd_mask   ( a.data.reg, b.data.reg, PRED_F ) ); \
    ASIMD_PLUS_CMP( AVX512, TAG, SI32, 16, 1, MASK_REGISTER, _mm512_cmp_epi32_mask( a.data.reg, b.data.reg, PRED_I ) ); \
    ASIMD_PLUS_CMP( AVX512, TAG, PI32, 16, 1, MASK_REGISTER, _mm512_cmp_epu32_mask( a.data.reg, b.data.reg, PRED_I ) ); \
    ASIMD_PLUS_CMP( AVX512, TAG, SI64,  8, 1, MASK_REGISTER, _mm512_cmp_epi64_mask( a.data.reg, b.data.reg, PRED_I ) ); \
    ASIMD_PLUS_CMP( AVX512, TAG, PI64,  8, 1, MASK_REGISTER, _mm512_cmp_epu64_mask( a.data.reg, b.data.reg, PRED_I ) )

ASIMD_PLUS_AVX512_CMP( cmp_gt, _CMP_GT_OQ, _MM_CMPINT_NLE );
ASIMD_PLUS_AVX512_CMP( cmp_lt, _CMP_LT_OQ, _MM_CMPINT_LT  );
ASIMD_PLUS_AVX512_CMP( cmp_eq, _CMP_EQ_OQ, _MM_CMPINT_EQ  );
ASIMD_PLUS_AVX512_CMP( cmp_ge, _CMP_GE_OQ, _MM_CMPINT_NLT );

#undef ASIMD_PLUS_AVX512_CMP

// ---- select driven by a mask register: a masked move, not a blend.
ASIMD_PLUS_SELECT( AVX512, FP32, 16, 1, MASK_REGISTER, _mm512_mask_blend_ps   ( m.data.reg, b.data.reg, a.data.reg ) );
ASIMD_PLUS_SELECT( AVX512, FP64,  8, 1, MASK_REGISTER, _mm512_mask_blend_pd   ( m.data.reg, b.data.reg, a.data.reg ) );
ASIMD_PLUS_SELECT( AVX512, SI32, 16, 1, MASK_REGISTER, _mm512_mask_blend_epi32( m.data.reg, b.data.reg, a.data.reg ) );
ASIMD_PLUS_SELECT( AVX512, PI32, 16, 1, MASK_REGISTER, _mm512_mask_blend_epi32( m.data.reg, b.data.reg, a.data.reg ) );
ASIMD_PLUS_SELECT( AVX512, SI64,  8, 1, MASK_REGISTER, _mm512_mask_blend_epi64( m.data.reg, b.data.reg, a.data.reg ) );
ASIMD_PLUS_SELECT( AVX512, PI64,  8, 1, MASK_REGISTER, _mm512_mask_blend_epi64( m.data.reg, b.data.reg, a.data.reg ) );

// ---- to_bits / mask_from_bits: a `k` register IS the integer, so both are a `kmov`.
ASIMD_PLUS_TO_BITS( AVX512, 16, 1, MASK_REGISTER, PI64( m.data.reg ) );
ASIMD_PLUS_TO_BITS( AVX512,  8, 1, MASK_REGISTER, PI64( m.data.reg ) );
ASIMD_PLUS_TO_BITS( AVX512,  4, 1, MASK_REGISTER, PI64( m.data.reg ) & 0xfu );
ASIMD_PLUS_TO_BITS( AVX512,  2, 1, MASK_REGISTER, PI64( m.data.reg ) & 0x3u );

ASIMD_PLUS_MASK_FROM_BITS( AVX512, 16, 1, MASK_REGISTER, __mmask16( b ) );
ASIMD_PLUS_MASK_FROM_BITS( AVX512,  8, 1, MASK_REGISTER, __mmask8 ( b ) );
ASIMD_PLUS_MASK_FROM_BITS( AVX512,  4, 1, MASK_REGISTER, __mmask8 ( b & 0xfu ) );
ASIMD_PLUS_MASK_FROM_BITS( AVX512,  2, 1, MASK_REGISTER, __mmask8 ( b & 0x3u ) );

// ---- fma, permute, bcast at 512 bits. `vpermps` becomes `vpermt2ps`-class here: one shot over
// sixteen lanes, where the generic form was sixteen loads through memory.
ASIMD_PLUS_FMA( AVX512, AVX512, FP32, 16, _mm512_fmadd_ps );
ASIMD_PLUS_FMA( AVX512, AVX512, FP64,  8, _mm512_fmadd_pd );

ASIMD_PLUS_PERMUTE( AVX512, FP32, 16, _mm512_permutexvar_ps   ( idx.data.reg, v.data.reg ) );
ASIMD_PLUS_PERMUTE( AVX512, SI32, 16, _mm512_permutexvar_epi32( idx.data.reg, v.data.reg ) );
ASIMD_PLUS_PERMUTE( AVX512, PI32, 16, _mm512_permutexvar_epi32( idx.data.reg, v.data.reg ) );
// the index vector is `SI32 x 8`, i.e. a __m256i: `vpermpd` reads 64-bit indices, so it is
// widened first. One `vpmovsxdq` plus one `vpermpd`, against eight loads through memory.
ASIMD_PLUS_PERMUTE( AVX512, FP64,  8, _mm512_permutexvar_pd   ( _mm512_cvtepi32_epi64( idx.data.reg ), v.data.reg ) );
ASIMD_PLUS_PERMUTE( AVX512, SI64,  8, _mm512_permutexvar_epi64( _mm512_cvtepi32_epi64( idx.data.reg ), v.data.reg ) );
ASIMD_PLUS_PERMUTE( AVX512, PI64,  8, _mm512_permutexvar_epi64( _mm512_cvtepi32_epi64( idx.data.reg ), v.data.reg ) );

ASIMD_PLUS_BCAST( AVX512, FP32, 16, _mm512_permutexvar_ps   ( _mm512_set1_epi32( LANE ), v.data.reg ) );
ASIMD_PLUS_BCAST( AVX512, SI32, 16, _mm512_permutexvar_epi32( _mm512_set1_epi32( LANE ), v.data.reg ) );
ASIMD_PLUS_BCAST( AVX512, FP64,  8, _mm512_permutexvar_pd   ( _mm512_set1_epi64( LANE ), v.data.reg ) );
ASIMD_PLUS_BCAST( AVX512, SI64,  8, _mm512_permutexvar_epi64( _mm512_set1_epi64( LANE ), v.data.reg ) );
ASIMD_PLUS_BCAST( AVX512, PI32, 16, _mm512_permutexvar_epi32( _mm512_set1_epi32( LANE ), v.data.reg ) );
ASIMD_PLUS_BCAST( AVX512, PI64,  8, _mm512_permutexvar_epi64( _mm512_set1_epi64( LANE ), v.data.reg ) );

#ifdef ASIMD_X86_HAS_AVX512VL
// ---- THE SAME, AT 128 AND 256 BITS. This is what AVX-512VL is: the AVX-512 encodings, hence the
// mask registers, on the narrower widths. Without these, a comparison at eight lanes on an
// AVX-512 machine still produced a 256-bit LANE mask and `select` still emitted `vblendvps` --
// measured at 3.07 ns per cut against 2.59 for intrinsics on the clipping kernel.
#define ASIMD_PLUS_VL_CMP( TAG, PRED_F, PRED_I ) \
    ASIMD_PLUS_CMP( AVX512VL, TAG, FP32, 8, 1, MASK_REGISTER, _mm256_cmp_ps_mask   ( a.data.reg, b.data.reg, PRED_F ) ); \
    ASIMD_PLUS_CMP( AVX512VL, TAG, FP64, 4, 1, MASK_REGISTER, _mm256_cmp_pd_mask   ( a.data.reg, b.data.reg, PRED_F ) ); \
    ASIMD_PLUS_CMP( AVX512VL, TAG, SI32, 8, 1, MASK_REGISTER, _mm256_cmp_epi32_mask( a.data.reg, b.data.reg, PRED_I ) ); \
    ASIMD_PLUS_CMP( AVX512VL, TAG, PI32, 8, 1, MASK_REGISTER, _mm256_cmp_epu32_mask( a.data.reg, b.data.reg, PRED_I ) ); \
    ASIMD_PLUS_CMP( AVX512VL, TAG, SI64, 4, 1, MASK_REGISTER, _mm256_cmp_epi64_mask( a.data.reg, b.data.reg, PRED_I ) ); \
    ASIMD_PLUS_CMP( AVX512VL, TAG, PI64, 4, 1, MASK_REGISTER, _mm256_cmp_epu64_mask( a.data.reg, b.data.reg, PRED_I ) ); \
    ASIMD_PLUS_CMP( AVX512VL, TAG, FP32, 4, 1, MASK_REGISTER, _mm_cmp_ps_mask      ( a.data.reg, b.data.reg, PRED_F ) ); \
    ASIMD_PLUS_CMP( AVX512VL, TAG, FP64, 2, 1, MASK_REGISTER, _mm_cmp_pd_mask      ( a.data.reg, b.data.reg, PRED_F ) ); \
    ASIMD_PLUS_CMP( AVX512VL, TAG, SI32, 4, 1, MASK_REGISTER, _mm_cmp_epi32_mask   ( a.data.reg, b.data.reg, PRED_I ) ); \
    ASIMD_PLUS_CMP( AVX512VL, TAG, PI32, 4, 1, MASK_REGISTER, _mm_cmp_epu32_mask   ( a.data.reg, b.data.reg, PRED_I ) ); \
    ASIMD_PLUS_CMP( AVX512VL, TAG, SI64, 2, 1, MASK_REGISTER, _mm_cmp_epi64_mask   ( a.data.reg, b.data.reg, PRED_I ) ); \
    ASIMD_PLUS_CMP( AVX512VL, TAG, PI64, 2, 1, MASK_REGISTER, _mm_cmp_epu64_mask   ( a.data.reg, b.data.reg, PRED_I ) )

ASIMD_PLUS_VL_CMP( cmp_gt, _CMP_GT_OQ, _MM_CMPINT_NLE );
ASIMD_PLUS_VL_CMP( cmp_lt, _CMP_LT_OQ, _MM_CMPINT_LT  );
ASIMD_PLUS_VL_CMP( cmp_eq, _CMP_EQ_OQ, _MM_CMPINT_EQ  );
ASIMD_PLUS_VL_CMP( cmp_ge, _CMP_GE_OQ, _MM_CMPINT_NLT );

#undef ASIMD_PLUS_VL_CMP

#define ASIMD_PLUS_VL_SELECT( T, N, FUNC ) \
    ASIMD_PLUS_SELECT( AVX512VL, T, N, 1, MASK_REGISTER, FUNC( m.data.reg, b.data.reg, a.data.reg ) )
ASIMD_PLUS_VL_SELECT( FP32, 8, _mm256_mask_blend_ps    );
ASIMD_PLUS_VL_SELECT( FP64, 4, _mm256_mask_blend_pd    );
ASIMD_PLUS_VL_SELECT( SI32, 8, _mm256_mask_blend_epi32 );
ASIMD_PLUS_VL_SELECT( PI32, 8, _mm256_mask_blend_epi32 );
ASIMD_PLUS_VL_SELECT( SI64, 4, _mm256_mask_blend_epi64 );
ASIMD_PLUS_VL_SELECT( PI64, 4, _mm256_mask_blend_epi64 );
ASIMD_PLUS_VL_SELECT( FP32, 4, _mm_mask_blend_ps       );
ASIMD_PLUS_VL_SELECT( FP64, 2, _mm_mask_blend_pd       );
ASIMD_PLUS_VL_SELECT( SI32, 4, _mm_mask_blend_epi32    );
ASIMD_PLUS_VL_SELECT( PI32, 4, _mm_mask_blend_epi32    );
ASIMD_PLUS_VL_SELECT( SI64, 2, _mm_mask_blend_epi64    );
ASIMD_PLUS_VL_SELECT( PI64, 2, _mm_mask_blend_epi64    );
#undef ASIMD_PLUS_VL_SELECT

// a full 4-lane permutation of 64-bit elements needs `vpermpd` with a REGISTER control, which is
// AVX-512VL: the AVX2 `_mm256_permutevar_pd` only moves within each 128-bit half.
ASIMD_PLUS_PERMUTE( AVX512VL, FP64, 4, _mm256_permutexvar_pd   ( _mm256_cvtepi32_epi64( idx.data.reg ), v.data.reg ) );
ASIMD_PLUS_PERMUTE( AVX512VL, SI64, 4, _mm256_permutexvar_epi64( _mm256_cvtepi32_epi64( idx.data.reg ), v.data.reg ) );
#endif // ASIMD_X86_HAS_AVX512VL

#endif // ASIMD_X86_HAS_AVX512F

} // namespace asimd

#endif // x86
