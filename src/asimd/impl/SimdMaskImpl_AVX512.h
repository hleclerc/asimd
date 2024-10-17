#pragma once

//#include "SimdVecImpl_Generic.h"
#include <x86intrin.h>

namespace asimd {
namespace internal {

#ifdef __AVX512F__

// struct Impl<...>
// SIMD_VEC_IMPL_REG( AVX512, std::uint64_t,  8, __m512i );
// SIMD_VEC_IMPL_REG( AVX512, std::int64_t ,  8, __m512i );
// SIMD_VEC_IMPL_REG( AVX512, double       ,  8, __m512d );
// SIMD_VEC_IMPL_REG( AVX512, std::uint32_t, 16, __m512i );
// SIMD_VEC_IMPL_REG( AVX512, std::int32_t , 16, __m512i );
// SIMD_VEC_IMPL_REG( AVX512, float        , 16, __m512  );

#endif // __AVX512F__

} // namespace internal
} // namespace asimd
