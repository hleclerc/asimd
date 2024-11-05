#pragma once

#include "../architectures/X86CpuFeatures.h"
#include "SimdVecImpl_Generic.h"
#include <x86intrin.h>

namespace asimd {
namespace internal {

#ifdef __AVX__


// struct Impl<...>
SIMD_VEC_IMPL_REG( AVX, PI64, 4, __m256i ); 
SIMD_VEC_IMPL_REG( AVX, SI64, 4, __m256i );
SIMD_VEC_IMPL_REG( AVX, FP64, 4, __m256d );
SIMD_VEC_IMPL_REG( AVX, PI32, 8, __m256i );
SIMD_VEC_IMPL_REG( AVX, SI32, 8, __m256i );
SIMD_VEC_IMPL_REG( AVX, FP32, 8, __m256  );

// init ----------------------------------------------------------------------
SIMD_VEC_IMPL_REG_INIT_1( AVX, PI64, 4, _mm256_set1_epi64x( a ) );
SIMD_VEC_IMPL_REG_INIT_1( AVX, SI64, 4, _mm256_set1_epi64x( a ) );
SIMD_VEC_IMPL_REG_INIT_1( AVX, FP64, 4, _mm256_set1_pd( a ) );
SIMD_VEC_IMPL_REG_INIT_1( AVX, PI32, 8, _mm256_set1_epi32( a ) );
SIMD_VEC_IMPL_REG_INIT_1( AVX, SI32, 8, _mm256_set1_epi32( a ) );
SIMD_VEC_IMPL_REG_INIT_1( AVX, FP32, 8, _mm256_set1_ps( a ) );

SIMD_VEC_IMPL_REG_INIT_4( AVX, PI64, 4, _mm256_set_epi64x( d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_4( AVX, SI64, 4, _mm256_set_epi64x( d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_4( AVX, FP64, 4, _mm256_set_pd( d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_8( AVX, PI32, 8, _mm256_set_epi32( h, g, f, e, d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_8( AVX, SI32, 8, _mm256_set_epi32( h, g, f, e, d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_8( AVX, FP32, 8, _mm256_set_ps( h, g, f, e, d, c, b, a ) );

// load_aligned -----------------------------------------------------------------------
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( AVX, PI64, 4, 256, _mm256_load_si256( (const __m256i *)data ) );  
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( AVX, SI64, 4, 256, _mm256_load_si256( (const __m256i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( AVX, FP64, 4, 256, _mm256_load_pd   (                  data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( AVX, PI32, 8, 256, _mm256_load_si256( (const __m256i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( AVX, SI32, 8, 256, _mm256_load_si256( (const __m256i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( AVX, FP32, 8, 256, _mm256_load_ps   (                  data ) );

SIMD_VEC_IMPL_REG_LOAD_ALIGNED_STREAM( AVX, PI64, 4, 256,         _mm256_stream_load_si256( (const __m256i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED_STREAM( AVX, SI64, 4, 256,         _mm256_stream_load_si256( (const __m256i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED_STREAM( AVX, FP64, 4, 256, (__m256)_mm256_stream_load_si256( (const __m256i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED_STREAM( AVX, PI32, 8, 256,         _mm256_stream_load_si256( (const __m256i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED_STREAM( AVX, SI32, 8, 256,         _mm256_stream_load_si256( (const __m256i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED_STREAM( AVX, FP32, 8, 256, (__m256)_mm256_stream_load_si256( (const __m256i *)data ) );

// load unaligned ----------------------------------------------------------------------
SIMD_VEC_IMPL_REG_LOAD_UNALIGNED( AVX, PI64, 4, _mm256_loadu_si256( (const __m256i *)data ) ); 
SIMD_VEC_IMPL_REG_LOAD_UNALIGNED( AVX, SI64, 4, _mm256_loadu_si256( (const __m256i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_UNALIGNED( AVX, FP64, 4, _mm256_loadu_pd   (                  data ) );
SIMD_VEC_IMPL_REG_LOAD_UNALIGNED( AVX, PI32, 8, _mm256_loadu_si256( (const __m256i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_UNALIGNED( AVX, SI32, 8, _mm256_loadu_si256( (const __m256i *)data ) ); 
SIMD_VEC_IMPL_REG_LOAD_UNALIGNED( AVX, FP32, 8, _mm256_loadu_ps   (                  data ) );

// store_aligned -----------------------------------------------------------------------
SIMD_VEC_IMPL_REG_STORE_ALIGNED( AVX, PI64, 4, 256, _mm256_store_si256( (__m256i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( AVX, SI64, 4, 256, _mm256_store_si256( (__m256i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( AVX, FP64, 4, 256, _mm256_store_pd   (            data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( AVX, PI32, 8, 256, _mm256_store_si256( (__m256i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( AVX, SI32, 8, 256, _mm256_store_si256( (__m256i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( AVX, FP32, 8, 256, _mm256_store_ps   (            data, impl.data.reg ) );

SIMD_VEC_IMPL_REG_STORE_ALIGNED_STREAM( AVX, PI64, 4, 256, _mm256_stream_si256( (__m256i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED_STREAM( AVX, SI64, 4, 256, _mm256_stream_si256( (__m256i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED_STREAM( AVX, FP64, 4, 256, _mm256_stream_pd   (            data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED_STREAM( AVX, PI32, 8, 256, _mm256_stream_si256( (__m256i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED_STREAM( AVX, SI32, 8, 256, _mm256_stream_si256( (__m256i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED_STREAM( AVX, FP32, 8, 256, _mm256_stream_ps   (            data, impl.data.reg ) );

// store unaligned ---------------------------------------------------------------------
SIMD_VEC_IMPL_REG_STORE_UNALIGNED( AVX, PI64, 4, _mm256_storeu_si256( (__m256i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_UNALIGNED( AVX, SI64, 4, _mm256_storeu_si256( (__m256i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_UNALIGNED( AVX, FP64, 4, _mm256_storeu_pd   (            data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_UNALIGNED( AVX, PI32, 8, _mm256_storeu_si256( (__m256i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_UNALIGNED( AVX, SI32, 8, _mm256_storeu_si256( (__m256i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_UNALIGNED( AVX, FP32, 8, _mm256_storeu_ps   (            data, impl.data.reg ) );

//// arithmetic operations that work on all types ------------------------------------------------
#define SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX_A( NAME ) \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX, PI64, 4, NAME, _mm256_##NAME##_epi64 ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX, SI64, 4, NAME, _mm256_##NAME##_epi64 ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX, FP64, 4, NAME, _mm256_##NAME##_pd    ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX, PI32, 8, NAME, _mm256_##NAME##_epi32 ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX, SI32, 8, NAME, _mm256_##NAME##_epi32 ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX, FP32, 8, NAME, _mm256_##NAME##_ps    );

SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX_A( add );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX_A( sub );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX_A( min );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX_A( max );

#undef SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX_A

//// arithmetic operations that work only on float types ------------------------------------------------
#define SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX_F( NAME ) \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX, double, 4, NAME, _mm256_##NAME##_pd ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX, float , 8, NAME, _mm256_##NAME##_ps );

SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX_F( mul );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX_F( div );

#undef SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX_F

//// arithmetic operations with != func and name -------------------------------------------------------
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX , FP64, 4, anb, _mm256_and_pd    );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX , FP32, 8, anb, _mm256_and_ps    );

// cmp simdvec ---------------------------------------------------------------------------------------------
#define SIMD_VEC_IMPL_CMP_OP_SIMDVEC_AVX( NAME, CMP ) \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX, PI64, 4, 64, NAME, (__m256i)_mm256_cmp_epi64( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX, SI64, 4, 64, NAME, (__m256i)_mm256_cmp_epi64( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX, FP64, 4, 64, NAME, (__m256i)_mm256_cmp_pd   ( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX, PI32, 8, 32, NAME, (__m256i)_mm256_cmp_epi64( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX, SI32, 8, 32, NAME, (__m256i)_mm256_cmp_epi64( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX, FP32, 8, 32, NAME, (__m256i)_mm256_cmp_ps   ( a.data.reg, b.data.reg, CMP ) ); \

SIMD_VEC_IMPL_CMP_OP_SIMDVEC_AVX( lt, _CMP_LT_OS )
SIMD_VEC_IMPL_CMP_OP_SIMDVEC_AVX( gt, _CMP_GT_OS )

#undef SIMD_VEC_IMPL_CMP_OP_SIMDVEC_AVX

#endif // __AVX__

} // namespace internal
} // namespace asimd
