#pragma once

#ifdef __AVX2__

#include "SimdVecImpl_Generic.h"
#include "SimdVecImpl_AVX.h"
#include <x86intrin.h>

namespace asimd {
namespace internal {

//// arithmetic operations with != func and name -------------------------------------------------------
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, PI64, 4, anb, _mm256_and_si256 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, SI64, 4, anb, _mm256_and_si256 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, PI32, 8, anb, _mm256_and_si256 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, SI32, 8, anb, _mm256_and_si256 );


//// arithmetic operations that work only on int types -------------------------------------------------
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, PI64, 4, sll, _mm256_sllv_epi64 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, SI64, 4, sll, _mm256_sllv_epi64 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, PI32, 8, sll, _mm256_sllv_epi32 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, SI32, 8, sll, _mm256_sllv_epi32 );

// => SSE2 size
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, PI64, 2, sll, _mm_sllv_epi64 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, SI64, 2, sll, _mm_sllv_epi64 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, PI32, 4, sll, _mm_sllv_epi32 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, SI32, 4, sll, _mm_sllv_epi32 );

// gather -----------------------------------------------------------------------------------------------
SIMD_VEC_IMPL_REG_GATHER( AVX2, PI64, PI32, 4, _mm256_i32gather_epi64( (const SI64*)data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, SI64, PI32, 4, _mm256_i32gather_epi64( data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, FP64, PI32, 4, _mm256_i32gather_pd   ( data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, PI32, PI32, 8, _mm256_i32gather_epi32( (const SI32*)data, ind.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, SI32, PI32, 8, _mm256_i32gather_epi32( data, ind.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, FP32, PI32, 8, _mm256_i32gather_ps   ( data, ind.data.reg, 4 ) );

SIMD_VEC_IMPL_REG_GATHER( AVX2, PI64, SI32, 4, _mm256_i32gather_epi64( (const SI64*)data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, SI64, SI32, 4, _mm256_i32gather_epi64( data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, FP64, SI32, 4, _mm256_i32gather_pd   ( data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, PI32, SI32, 8, _mm256_i32gather_epi32( (const SI32*)data, ind.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, SI32, SI32, 8, _mm256_i32gather_epi32( data, ind.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, FP32, SI32, 8, _mm256_i32gather_ps   ( data, ind.data.reg, 4 ) );

// SSE2 sizes
SIMD_VEC_IMPL_REG_GATHER( AVX2, PI64, PI32, 2, _mm_i32gather_epi64   ( (const SI64*)data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, SI64, PI32, 2, _mm_i32gather_epi64   ( data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, FP64, PI32, 2, _mm_i32gather_pd      ( data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, PI32, PI32, 4, _mm_i32gather_epi32   ( (const SI32*)data, ind.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, SI32, PI32, 4, _mm_i32gather_epi32   ( data, ind.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, FP32, PI32, 4, _mm_i32gather_ps      ( data, ind.data.reg, 4 ) );
   
SIMD_VEC_IMPL_REG_GATHER( AVX2, PI64, SI32, 2, _mm_i32gather_epi64   ( (const SI64*)data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, SI64, SI32, 2, _mm_i32gather_epi64   ( data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, FP64, SI32, 2, _mm_i32gather_pd      ( data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, PI32, SI32, 4, _mm_i32gather_epi32   ( (const SI32*)data, ind.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, SI32, SI32, 4, _mm_i32gather_epi32   ( data, ind.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, FP32, SI32, 4, _mm_i32gather_ps      ( data, ind.data.reg, 4 ) );


} // namespace internal
} // namespace asimd

#endif // __AVX2__