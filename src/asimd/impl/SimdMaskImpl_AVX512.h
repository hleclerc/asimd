#pragma once

#include "../architectures/X86CpuFeatures.h"
#include "SimdMaskImpl_Generic.h"
#include <x86intrin.h>

namespace asimd {
namespace internal {

#ifdef __AVX512F__

// struct SimdMaskImpl<...>
SIMD_MASK_IMPL_REG_BITS_UNSPLITABLE( AVX512,  8, __mmask8  )
SIMD_MASK_IMPL_REG_BITS_SPLITABLE  ( AVX512, 16, __mmask16 )
SIMD_MASK_IMPL_REG_BITS_SPLITABLE  ( AVX512, 32, __mmask32 )
SIMD_MASK_IMPL_REG_BITS_SPLITABLE  ( AVX512, 64, __mmask64 )

SIMD_MASK_IMPL_REG_REDUCTION( AVX512,  8, 1, all, ~mask.data.reg == 0 ) 
SIMD_MASK_IMPL_REG_REDUCTION( AVX512, 16, 1, all, ~mask.data.reg == 0 ) 
SIMD_MASK_IMPL_REG_REDUCTION( AVX512, 32, 1, all, ~mask.data.reg == 0 ) 
SIMD_MASK_IMPL_REG_REDUCTION( AVX512, 64, 1, all, ~mask.data.reg == 0 ) 

SIMD_MASK_IMPL_REG_REDUCTION( AVX512,  8, 1, any, mask.data.reg != 0 ) 
SIMD_MASK_IMPL_REG_REDUCTION( AVX512, 16, 1, any, mask.data.reg != 0 ) 
SIMD_MASK_IMPL_REG_REDUCTION( AVX512, 32, 1, any, mask.data.reg != 0 ) 
SIMD_MASK_IMPL_REG_REDUCTION( AVX512, 64, 1, any, mask.data.reg != 0 ) 

#endif // __AVX512F__

} // namespace internal
} // namespace asimd
