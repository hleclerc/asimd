#pragma once
 
#ifdef __AVX512F__

#include "../architectures/X86CpuFeatures.h"
#include "SimdVecImpl_Generic.h"
#include <immintrin.h>
#include <x86intrin.h>

namespace asimd {
namespace internal {
 

// struct Impl<...>
SIMD_VEC_IMPL_REG( AVX512, PI64,  8, __m512i );
SIMD_VEC_IMPL_REG( AVX512, SI64,  8, __m512i );
SIMD_VEC_IMPL_REG( AVX512, FP64,  8, __m512d );
SIMD_VEC_IMPL_REG( AVX512, PI32, 16, __m512i );
SIMD_VEC_IMPL_REG( AVX512, SI32, 16, __m512i );
SIMD_VEC_IMPL_REG( AVX512, FP32, 16, __m512  );

// init ----------------------------------------------------------------------
SIMD_VEC_IMPL_REG_INIT_1( AVX512, PI64,  8, _mm512_set1_epi64( a ) );
SIMD_VEC_IMPL_REG_INIT_1( AVX512, SI64,  8, _mm512_set1_epi64( a ) );
SIMD_VEC_IMPL_REG_INIT_1( AVX512, FP64,  8, _mm512_set1_pd   ( a ) );
SIMD_VEC_IMPL_REG_INIT_1( AVX512, PI32, 16, _mm512_set1_epi32( a ) );
SIMD_VEC_IMPL_REG_INIT_1( AVX512, SI32, 16, _mm512_set1_epi32( a ) );
SIMD_VEC_IMPL_REG_INIT_1( AVX512, FP32, 16, _mm512_set1_ps   ( a ) );

SIMD_VEC_IMPL_REG_INIT_8 ( AVX512, PI64,  8, _mm512_set_epi64( h, g, f, e, d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_8 ( AVX512, SI64,  8, _mm512_set_epi64( h, g, f, e, d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_8 ( AVX512, FP64,  8, _mm512_set_pd   ( h, g, f, e, d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_16( AVX512, PI32, 16, _mm512_set_epi32( p, o, n, m, l, k, j, i, h, g, f, e, d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_16( AVX512, SI32, 16, _mm512_set_epi32( p, o, n, m, l, k, j, i, h, g, f, e, d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_16( AVX512, FP32, 16, _mm512_set_ps   ( p, o, n, m, l, k, j, i, h, g, f, e, d, c, b, a ) );

// load_aligned -----------------------------------------------------------------------
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( AVX512, PI64,  8, 512, _mm512_load_si512( (const __m512i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( AVX512, SI64,  8, 512, _mm512_load_si512( (const __m512i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( AVX512, FP64,  8, 512, _mm512_load_pd   (                  data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( AVX512, PI32, 16, 512, _mm512_load_si512( (const __m512i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( AVX512, SI32, 16, 512, _mm512_load_si512( (const __m512i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( AVX512, FP32, 16, 512, _mm512_load_ps   (                  data ) );

SIMD_VEC_IMPL_REG_LOAD_ALIGNED_STREAM( AVX512, PI64,  8, 512,         _mm512_stream_load_si512( (const __m512i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED_STREAM( AVX512, SI64,  8, 512,         _mm512_stream_load_si512( (const __m512i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED_STREAM( AVX512, FP64,  8, 512, (__m512)_mm512_stream_load_si512( (const __m512i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED_STREAM( AVX512, PI32, 16, 512,         _mm512_stream_load_si512( (const __m512i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED_STREAM( AVX512, SI32, 16, 512,         _mm512_stream_load_si512( (const __m512i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED_STREAM( AVX512, FP32, 16, 512, (__m512)_mm512_stream_load_si512( (const __m512i *)data ) );

// load -------------------------------------------------------------------------------
SIMD_VEC_IMPL_REG_LOAD_UNALIGNED( AVX512, PI64,  8, _mm512_loadu_si512( (const __m512i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_UNALIGNED( AVX512, SI64,  8, _mm512_loadu_si512( (const __m512i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_UNALIGNED( AVX512, FP64,  8, _mm512_loadu_pd   (                  data ) );
SIMD_VEC_IMPL_REG_LOAD_UNALIGNED( AVX512, PI32, 16, _mm512_loadu_si512( (const __m512i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_UNALIGNED( AVX512, SI32, 16, _mm512_loadu_si512( (const __m512i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_UNALIGNED( AVX512, FP32, 16, _mm512_loadu_ps   (                  data ) );

// store_aligned -----------------------------------------------------------------------
SIMD_VEC_IMPL_REG_STORE_ALIGNED( AVX512, PI64,  8, 512, _mm512_store_si512( (__m512i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( AVX512, SI64,  8, 512, _mm512_store_si512( (__m512i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( AVX512, FP64,  8, 512, _mm512_store_pd   (            data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( AVX512, PI32, 16, 512, _mm512_store_si512( (__m512i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( AVX512, SI32, 16, 512, _mm512_store_si512( (__m512i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( AVX512, FP32, 16, 512, _mm512_store_ps   (            data, impl.data.reg ) );

SIMD_VEC_IMPL_REG_STORE_ALIGNED_STREAM( AVX512, PI64,  8, 512, _mm512_stream_si512( (__m512i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED_STREAM( AVX512, SI64,  8, 512, _mm512_stream_si512( (__m512i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED_STREAM( AVX512, FP64,  8, 512, _mm512_stream_pd   (            data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED_STREAM( AVX512, PI32, 16, 512, _mm512_stream_si512( (__m512i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED_STREAM( AVX512, SI32, 16, 512, _mm512_stream_si512( (__m512i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED_STREAM( AVX512, FP32, 16, 512, _mm512_stream_ps   (            data, impl.data.reg ) );

// store -----------------------------------------------------------------------
SIMD_VEC_IMPL_REG_STORE_UNALIGNED( AVX512, PI64,  8, _mm512_storeu_si512( (__m512i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_UNALIGNED( AVX512, SI64,  8, _mm512_storeu_si512( (__m512i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_UNALIGNED( AVX512, FP64,  8, _mm512_storeu_pd   (            data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_UNALIGNED( AVX512, PI32, 16, _mm512_storeu_si512( (__m512i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_UNALIGNED( AVX512, SI32, 16, _mm512_storeu_si512( (__m512i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_UNALIGNED( AVX512, FP32, 16, _mm512_storeu_ps   (            data, impl.data.reg ) );

//// arithmetic operations that work on all types ------------------------------------------------
#define SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_A( NAME ) \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, PI64,  8, NAME, _mm512_##NAME##_epi64 ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, SI64,  8, NAME, _mm512_##NAME##_epi64 ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, FP64,  8, NAME, _mm512_##NAME##_pd    ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, PI32, 16, NAME, _mm512_##NAME##_epi32 ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, SI32, 16, NAME, _mm512_##NAME##_epi32 ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, FP32, 16, NAME, _mm512_##NAME##_ps    );

    SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_A( add );
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_A( sub );
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_A( min );
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_A( max );

#undef SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_A

//// arithmetic operations that work only on float types ------------------------------------------------
#define SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_F( NAME ) \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, double,  8, NAME, _mm512_##NAME##_pd ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, float , 16, NAME, _mm512_##NAME##_ps );

    SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_F( mul );
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_F( div );

#undef SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_F

//// arithmetic operations with != func and name -------------------------------------------------------
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, PI64,  8, anb, _mm512_and_si512 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, SI64,  8, anb, _mm512_and_si512 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, FP64,  8, anb, _mm512_and_pd    );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, PI32, 16, anb, _mm512_and_si512 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, SI32, 16, anb, _mm512_and_si512 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, FP32, 16, anb, _mm512_and_ps    );

//// arithmetic operations that work only on int types ------------------------------------------------
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, PI64,  8, sll, _mm512_sllv_epi64 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, SI64,  8, sll, _mm512_sllv_epi64 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, PI32, 16, sll, _mm512_sllv_epi32 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, SI32, 16, sll, _mm512_sllv_epi32 );

// gather -----------------------------------------------------------------------------------------------
SIMD_VEC_IMPL_REG_GATHER( AVX512, PI64, PI32,  8, _mm512_i32gather_epi64( ind.data.reg, data, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX512, SI64, PI32,  8, _mm512_i32gather_epi64( ind.data.reg, data, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX512, FP64, PI32,  8, _mm512_i32gather_pd   ( ind.data.reg, data, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX512, PI32, PI32, 16, _mm512_i32gather_epi32( ind.data.reg, data, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX512, SI32, PI32, 16, _mm512_i32gather_epi32( ind.data.reg, data, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX512, FP32, PI32, 16, _mm512_i32gather_ps   ( ind.data.reg, data, 4 ) );

SIMD_VEC_IMPL_REG_GATHER( AVX512, PI64, SI32,  8, _mm512_i32gather_epi64( ind.data.reg, data, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX512, SI64, SI32,  8, _mm512_i32gather_epi64( ind.data.reg, data, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX512, FP64, SI32,  8, _mm512_i32gather_pd   ( ind.data.reg, data, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX512, PI32, SI32, 16, _mm512_i32gather_epi32( ind.data.reg, data, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX512, SI32, SI32, 16, _mm512_i32gather_epi32( ind.data.reg, data, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX512, FP32, SI32, 16, _mm512_i32gather_ps   ( ind.data.reg, data, 4 ) );

// scatter -----------------------------------------------------------------------------------------------
SIMD_VEC_IMPL_REG_SCATTER( AVX512, PI64, PI32,  8, _mm512_i32scatter_epi64( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512, SI64, PI32,  8, _mm512_i32scatter_epi64( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512, FP64, PI32,  8, _mm512_i32scatter_pd   ( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512, PI32, PI32, 16, _mm512_i32scatter_epi32( data, ind.data.reg, vec.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512, SI32, PI32, 16, _mm512_i32scatter_epi32( data, ind.data.reg, vec.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512, FP32, PI32, 16, _mm512_i32scatter_ps   ( data, ind.data.reg, vec.data.reg, 4 ) );

SIMD_VEC_IMPL_REG_SCATTER( AVX512, PI64, SI32,  8, _mm512_i32scatter_epi64( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512, SI64, SI32,  8, _mm512_i32scatter_epi64( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512, FP64, SI32,  8, _mm512_i32scatter_pd   ( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512, PI32, SI32, 16, _mm512_i32scatter_epi32( data, ind.data.reg, vec.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512, SI32, SI32, 16, _mm512_i32scatter_epi32( data, ind.data.reg, vec.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512, FP32, SI32, 16, _mm512_i32scatter_ps   ( data, ind.data.reg, vec.data.reg, 4 ) );

// AVX2 sizes
SIMD_VEC_IMPL_REG_SCATTER( AVX512, PI64, PI32, 4, _mm256_i32scatter_epi64( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512, SI64, PI32, 4, _mm256_i32scatter_epi64( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512, FP64, PI32, 4, _mm256_i32scatter_pd   ( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512, PI32, PI32, 8, _mm256_i32scatter_epi32( data, ind.data.reg, vec.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512, SI32, PI32, 8, _mm256_i32scatter_epi32( data, ind.data.reg, vec.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512, FP32, PI32, 8, _mm256_i32scatter_ps   ( data, ind.data.reg, vec.data.reg, 4 ) );

SIMD_VEC_IMPL_REG_SCATTER( AVX512, PI64, SI32, 4, _mm256_i32scatter_epi64( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512, SI64, SI32, 4, _mm256_i32scatter_epi64( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512, FP64, SI32, 4, _mm256_i32scatter_pd   ( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512, PI32, SI32, 8, _mm256_i32scatter_epi32( data, ind.data.reg, vec.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512, SI32, SI32, 8, _mm256_i32scatter_epi32( data, ind.data.reg, vec.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512, FP32, SI32, 8, _mm256_i32scatter_ps   ( data, ind.data.reg, vec.data.reg, 4 ) );

// SSE2 sizes
SIMD_VEC_IMPL_REG_SCATTER( AVX512, PI64, PI32, 2, _mm_i32scatter_epi64( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512, SI64, PI32, 2, _mm_i32scatter_epi64( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512, FP64, PI32, 2, _mm_i32scatter_pd   ( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512, PI32, PI32, 4, _mm_i32scatter_epi32( data, ind.data.reg, vec.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512, SI32, PI32, 4, _mm_i32scatter_epi32( data, ind.data.reg, vec.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512, FP32, PI32, 4, _mm_i32scatter_ps   ( data, ind.data.reg, vec.data.reg, 4 ) );

SIMD_VEC_IMPL_REG_SCATTER( AVX512, PI64, SI32, 2, _mm_i32scatter_epi64( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512, SI64, SI32, 2, _mm_i32scatter_epi64( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512, FP64, SI32, 2, _mm_i32scatter_pd   ( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512, PI32, SI32, 4, _mm_i32scatter_epi32( data, ind.data.reg, vec.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512, SI32, SI32, 4, _mm_i32scatter_epi32( data, ind.data.reg, vec.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512, FP32, SI32, 4, _mm_i32scatter_ps   ( data, ind.data.reg, vec.data.reg, 4 ) );

// cmp simdvec ---------------------------------------------------------------------------------------------
#define SIMD_VEC_IMPL_CMP_OP_SIMDVEC_AVX512( NAME, FLAG_F, FLAG_I ) \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX512, PI64,  8, 1, NAME, _mm512_cmp_epu64_mask( a.data.reg, b.data.reg, FLAG_I ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX512, SI64,  8, 1, NAME, _mm512_cmp_epi64_mask( a.data.reg, b.data.reg, FLAG_I ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX512, FP64,  8, 1, NAME, _mm512_cmp_pd_mask   ( a.data.reg, b.data.reg, FLAG_F ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX512, PI32, 16, 1, NAME, _mm512_cmp_epu32_mask( a.data.reg, b.data.reg, FLAG_I ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX512, SI32, 16, 1, NAME, _mm512_cmp_epi32_mask( a.data.reg, b.data.reg, FLAG_I ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX512, FP32, 16, 1, NAME, _mm512_cmp_ps_mask   ( a.data.reg, b.data.reg, FLAG_F ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX512, PI16, 32, 1, NAME, _mm512_cmp_epu16_mask( a.data.reg, b.data.reg, FLAG_I ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX512, SI16, 32, 1, NAME, _mm512_cmp_epi16_mask( a.data.reg, b.data.reg, FLAG_I ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX512, PI8 , 64, 1, NAME, _mm512_cmp_epu8_mask ( a.data.reg, b.data.reg, FLAG_I ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX512, SI8 , 64, 1, NAME, _mm512_cmp_epi8_mask ( a.data.reg, b.data.reg, FLAG_I ) ); \

SIMD_VEC_IMPL_CMP_OP_SIMDVEC_AVX512( lt, _CMP_LT_OS, _MM_CMPINT_LT )
SIMD_VEC_IMPL_CMP_OP_SIMDVEC_AVX512( gt, _CMP_GT_OS, _MM_CMPINT_GT )

} // namespace internal
} // namespace asimd

#endif // __AVX512F__
