#pragma once

#include "X86Cpu.h"

namespace asimd {


// -------------------------- Native --------------------------
#if ( defined(_M_IX86) || defined(__i386__) || defined(_M_X64) || defined(__x86_64__) )
using NativeCpu = X86Cpu< 8 * sizeof( void * )
    #ifdef __AVX512F__
        , features::AVX512
    #endif
    #ifdef __AVX2__
        , features::AVX2
    #endif
    #ifdef __AVX__
       , features::AVX
    #endif
    #ifdef __SSE2__
       , features::SSE2
    #endif
    #ifdef __SSE__
       , features::SSE
    #endif
>;
#endif // x86


} // namespace asimd
