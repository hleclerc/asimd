#pragma once

#include "../architectures/X86CpuFeatures.h"
#include "SimdMaskImpl_Generic.h"
#include <x86intrin.h>

namespace asimd {
namespace internal {

#ifdef __SSE2__

// struct SimdMaskImpl<...>
SIMD_MASK_IMPL_REG_LARGE( SSE2, 2, 64, __m128i )
SIMD_MASK_IMPL_REG_LARGE( SSE2, 4, 32, __m128i )

SIMD_MASK_IMPL_REG_REDUCTION( SSE2, 2, 64, all, _mm_testz_si128( _mm_xor_si128( mask.data.reg, _mm_set1_epi32( -1 ) ), _mm_set1_epi32( -1 ) ) ) 
SIMD_MASK_IMPL_REG_REDUCTION( SSE2, 4, 32, all, _mm_testz_si128( _mm_xor_si128( mask.data.reg, _mm_set1_epi32( -1 ) ), _mm_set1_epi32( -1 ) ) ) 

SIMD_MASK_IMPL_REG_REDUCTION( SSE2, 2, 64, any, _mm_testz_si128( mask.data.reg, _mm_set1_epi32( -1 ) ) == 0 ) 
SIMD_MASK_IMPL_REG_REDUCTION( SSE2, 4, 32, any, _mm_testz_si128( mask.data.reg, _mm_set1_epi32( -1 ) ) == 0 ) 

#endif // __SSE2__

} // namespace internal
} // namespace asimd
