#pragma once

#include "../architectures/X86CpuFeatures.h"
#include "SimdVecImpl_Generic.h"
#include <x86intrin.h>

namespace asimd {
namespace internal {

#ifdef __SSE2__


// struct Impl<...>
SIMD_VEC_IMPL_REG( SSE2, PI64, 2, __m128i );
SIMD_VEC_IMPL_REG( SSE2, SI64, 2, __m128i );
SIMD_VEC_IMPL_REG( SSE2, FP64, 2, __m128d );
SIMD_VEC_IMPL_REG( SSE2, PI32, 4, __m128i );
SIMD_VEC_IMPL_REG( SSE2, SI32, 4, __m128i );
SIMD_VEC_IMPL_REG( SSE2, FP32, 4, __m128  );

// init -------------------------------------------------------------------------------
SIMD_VEC_IMPL_REG_INIT_1( SSE2, PI64, 2, _mm_set1_epi64x( a ) );
SIMD_VEC_IMPL_REG_INIT_1( SSE2, SI64, 2, _mm_set1_epi64x( a ) );
SIMD_VEC_IMPL_REG_INIT_1( SSE2, FP64, 2, _mm_set1_pd    ( a ) );
SIMD_VEC_IMPL_REG_INIT_1( SSE2, PI32, 4, _mm_set1_epi32 ( a ) );
SIMD_VEC_IMPL_REG_INIT_1( SSE2, SI32, 4, _mm_set1_epi32 ( a ) );
SIMD_VEC_IMPL_REG_INIT_1( SSE2, FP32, 4, _mm_set1_ps    ( a ) );

SIMD_VEC_IMPL_REG_INIT_2( SSE2, PI64, 2, _mm_set_epi64x( b, a ) );
SIMD_VEC_IMPL_REG_INIT_2( SSE2, SI64, 2, _mm_set_epi64x( b, a ) );
SIMD_VEC_IMPL_REG_INIT_2( SSE2, FP64, 2, _mm_set_pd    ( b, a ) );
SIMD_VEC_IMPL_REG_INIT_4( SSE2, PI32, 4, _mm_set_epi32 ( d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_4( SSE2, SI32, 4, _mm_set_epi32 ( d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_4( SSE2, FP32, 4, _mm_set_ps    ( d, c, b, a ) );

// prefetch ----------------------------------------------------------------------
template<int len,class Arch>
auto prefetch( const void *beg, N<len>, S<Arch> ) -> typename std::enable_if<(len <= 64) && Arch::template Has<features::SSE>::value>::type {
    ASIMD_DEBUG_ON_OP("prefetch","SSE","_mm_prefetch")
    _mm_prefetch( beg, _MM_HINT_ET0 );
}

template<int len,class Arch>
auto prefetch( const void *beg, N<len>, S<Arch> ) -> typename std::enable_if<(len > 64) && Arch::template Has<features::SSE>::value>::type {
    prefetch( reinterpret_cast<const char *>( beg ) + 0 * len / 2, N<len / 2>(), S<Arch>() );
    prefetch( reinterpret_cast<const char *>( beg ) + 1 * len / 2, N<len / 2>(), S<Arch>() );
}

// load_aligned -----------------------------------------------------------------------
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( SSE2, PI64, 2, 128, _mm_load_si128( (__m128i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( SSE2, SI64, 2, 128, _mm_load_si128( (__m128i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( SSE2, FP64, 2, 128, _mm_load_pd   (            data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( SSE2, PI32, 4, 128, _mm_load_si128( (__m128i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( SSE2, SI32, 4, 128, _mm_load_si128( (__m128i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( SSE2, FP32, 4, 128, _mm_load_ps   (            data ) );

SIMD_VEC_IMPL_REG_LOAD_ALIGNED_STREAM( SSE2, PI64, 2, 128,         _mm_stream_load_si128( (__m128i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED_STREAM( SSE2, SI64, 2, 128,         _mm_stream_load_si128( (__m128i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED_STREAM( SSE2, FP64, 2, 128, (__m128)_mm_stream_load_si128( (__m128i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED_STREAM( SSE2, PI32, 4, 128,         _mm_stream_load_si128( (__m128i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED_STREAM( SSE2, SI32, 4, 128,         _mm_stream_load_si128( (__m128i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED_STREAM( SSE2, FP32, 4, 128, (__m128)_mm_stream_load_si128( (__m128i *)data ) );

// load ------------------------------------------------------------------------------
SIMD_VEC_IMPL_REG_LOAD_UNALIGNED( SSE2, PI64, 2, _mm_loadu_si128( (__m128i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_UNALIGNED( SSE2, SI64, 2, _mm_loadu_si128( (__m128i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_UNALIGNED( SSE2, FP64, 2, _mm_loadu_pd   (            data ) );
SIMD_VEC_IMPL_REG_LOAD_UNALIGNED( SSE2, PI32, 4, _mm_loadu_si128( (__m128i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_UNALIGNED( SSE2, SI32, 4, _mm_loadu_si128( (__m128i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_UNALIGNED( SSE2, FP32, 4, _mm_loadu_ps   (            data ) );

// store_aligned -----------------------------------------------------------------------
SIMD_VEC_IMPL_REG_STORE_ALIGNED( SSE2, PI64, 2, 128, _mm_store_si128( (__m128i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( SSE2, SI64, 2, 128, _mm_store_si128( (__m128i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( SSE2, FP64, 2, 128, _mm_store_pd   (            data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( SSE2, PI32, 4, 128, _mm_store_si128( (__m128i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( SSE2, SI32, 4, 128, _mm_store_si128( (__m128i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( SSE2, FP32, 4, 128, _mm_store_ps   (            data, impl.data.reg ) );

SIMD_VEC_IMPL_REG_STORE_ALIGNED_STREAM( SSE2, PI64, 2, 128, _mm_stream_si128( (__m128i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED_STREAM( SSE2, SI64, 2, 128, _mm_stream_si128( (__m128i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED_STREAM( SSE2, FP64, 2, 128, _mm_stream_pd   (            data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED_STREAM( SSE2, PI32, 4, 128, _mm_stream_si128( (__m128i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED_STREAM( SSE2, SI32, 4, 128, _mm_stream_si128( (__m128i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED_STREAM( SSE2, FP32, 4, 128, _mm_stream_ps   (            data, impl.data.reg ) );

// store -------------------------------------------------------------------------------
SIMD_VEC_IMPL_REG_STORE_UNALIGNED( SSE2, PI64, 2, _mm_storeu_si128( (__m128i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_UNALIGNED( SSE2, SI64, 2, _mm_storeu_si128( (__m128i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_UNALIGNED( SSE2, FP64, 2, _mm_storeu_pd   (            data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_UNALIGNED( SSE2, PI32, 4, _mm_storeu_si128( (__m128i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_UNALIGNED( SSE2, SI32, 4, _mm_storeu_si128( (__m128i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_UNALIGNED( SSE2, FP32, 4, _mm_storeu_ps   (            data, impl.data.reg ) );

//// arithmetic operations that work on all types --------------------------------------
#define SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_A( NAME ) \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( SSE2, PI64, 2, NAME, _mm_##NAME##_epi64 ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( SSE2, SI64, 2, NAME, _mm_##NAME##_epi64 ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( SSE2, FP64, 2, NAME, _mm_##NAME##_pd    ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( SSE2, PI32, 4, NAME, _mm_##NAME##_epi32 ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( SSE2, SI32, 4, NAME, _mm_##NAME##_epi32 ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( SSE2, FP32, 4, NAME, _mm_##NAME##_ps    );

    SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_A( add );
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_A( sub );
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_A( min );
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_A( max );

#undef SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_A

//// arithmetic operations that work only on float types ------------------------------------------------
#define SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_F( NAME ) \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( SSE2, double, 2, NAME, _mm_##NAME##_pd ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( SSE2, float , 4, NAME, _mm_##NAME##_ps );

    SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_F( mul );
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_F( div );

#undef SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_F

//// arithmetic operations with != func and name -------------------------------------------------------
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( SSE2, PI64, 2, anb, _mm_and_si128 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( SSE2, SI64, 2, anb, _mm_and_si128 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( SSE2, FP64, 2, anb, _mm_and_pd    );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( SSE2, PI32, 4, anb, _mm_and_si128 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( SSE2, SI32, 4, anb, _mm_and_si128 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( SSE2, FP32, 4, anb, _mm_and_ps    );

//// cmp operations ------------------------------------------------------------------
#define SIMD_VEC_IMPL_CMP_OP_SIMDVEC_AVX( NAME, CMP ) \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( SSE2, PI64, PI64, 2, NAME, (__m128i)_mm_cmp_epi64( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( SSE2, SI64, PI64, 2, NAME, (__m128i)_mm_cmp_epi64( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( SSE2, FP64, PI64, 2, NAME, (__m128i)_mm_cmp_pd   ( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( SSE2, PI32, PI32, 4, NAME, (__m128i)_mm_cmp_epi64( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( SSE2, SI32, PI32, 4, NAME, (__m128i)_mm_cmp_epi64( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( SSE2, FP32, PI32, 4, NAME, (__m128i)_mm_cmp_ps   ( a.data.reg, b.data.reg, CMP ) ); \

SIMD_VEC_IMPL_CMP_OP_SIMDVEC_AVX( lt, _CMP_LT_OS )
SIMD_VEC_IMPL_CMP_OP_SIMDVEC_AVX( gt, _CMP_GT_OS )

#undef SIMD_VEC_IMPL_CMP_OP_SIMDVEC_AVX


#endif // __SSE2__

} // namespace internal
} // namespace asimd
