#pragma once

#include "X86Cpu.h"
#include "NativeCpu.h"

namespace asimd {


// -------------------------- Native --------------------------
#if ( defined(_M_IX86) || defined(__i386__) || defined(_M_X64) || defined(__x86_64__) )
using LargestCpu = X86Cpu< 8 * sizeof( void * ),
    features::AVX512, features::AVX512VL, features::AVX512BW, features::AVX512DQ,
    features::AVX2, features::AVX, features::FMA,
    features::SSE4_2, features::SSE4_1, features::SSSE3, features::SSE3,
    features::SSE2, features::SSE
>;
#else
using LargestCpu = NativeCpu; ///< no "largest" is known for this target; the native one will do.
#endif


} // namespace asimd
