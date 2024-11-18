#pragma once

#ifdef __AVX__

#include "../architectures/X86CpuFeatures.h"
#include "SimdMaskImpl_Generic.h"
#include <x86intrin.h>

namespace asimd {
namespace internal {


// struct Impl<...>
SIMD_MASK_IMPL_REG_LARGE( AVX, 4, 64, __m256i )
SIMD_MASK_IMPL_REG_LARGE( AVX, 8, 32, __m256i )

SIMD_MASK_IMPL_REG_REDUCTION( AVX, 4, 64, all, _mm256_testz_si256( _mm256_xor_si256( mask.data.reg, _mm256_set1_epi32( -1 ) ), _mm256_set1_epi32( -1 ) ) ) 
SIMD_MASK_IMPL_REG_REDUCTION( AVX, 8, 32, all, _mm256_testz_si256( _mm256_xor_si256( mask.data.reg, _mm256_set1_epi32( -1 ) ), _mm256_set1_epi32( -1 ) ) ) 

SIMD_MASK_IMPL_REG_REDUCTION( AVX, 4, 64, any, _mm256_testz_si256( mask.data.reg, _mm256_set1_epi32( -1 ) ) == 0 ) 
SIMD_MASK_IMPL_REG_REDUCTION( AVX, 8, 32, any, _mm256_testz_si256( mask.data.reg, _mm256_set1_epi32( -1 ) ) == 0 ) 

} // namespace internal
} // namespace asimd

#endif // __AVX__
