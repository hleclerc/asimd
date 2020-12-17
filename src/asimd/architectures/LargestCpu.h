#pragma once

#include "X86Cpu.h"

namespace asimd {


// -------------------------- Native --------------------------
#if ( defined(_M_IX86) || defined(__i386__) || defined(_M_X64) || defined(__x86_64__) )
using LargestCpu = X86Cpu< 8 * sizeof( void * ),
    features::AVX512,
    features::AVX2,
    features::AVX,
    features::SSE2
>;
#endif // x86


} // namespace asimd
