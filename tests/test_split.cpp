// THE DIFFERENTIATOR: a width that exceeds the target's register.
//
// `SimdVec<float,8>` on an architecture capped at SSE2 must split into two four-lane registers and
// give exactly the same answer. This is what Highway refuses at compile time ("Too many lanes"),
// and what will decide the NEON port.
#include <asimd/SimdOpsPlus.h>

#include "check.h"

using Sse    = asimd::X86Cpu<64, asimd::features::SSE2, asimd::features::SSE>;
using V8_sse = asimd::SimdVec<float,8,Sse>;              // eight lanes over four-lane registers
using V8_nat = asimd::SimdVec<float,8>;                  // eight lanes over one register, if any

int main() {
    CHECK( V8_sse::size() == 8 );
    CHECK( sizeof( V8_sse ) == 32 );

    alignas( 64 ) float src[ 8 ] = { 1, 2, 3, 4, 5, 6, 7, 8 };

    // the split must return what the native width returns
    {
        const float expected[ 8 ] = { 11, 12, 13, 14, 15, 16, 17, 18 };
        CHECK_LANES( expected, 8, V8_sse::load_aligned( src ) + V8_sse( 10.f ) );
        CHECK_LANES( expected, 8, V8_nat::load_aligned( src ) + V8_nat( 10.f ) );
    }
    {
        const float expected[ 8 ] = { 2, 4, 6, 8, 10, 12, 14, 16 };
        CHECK_LANES( expected, 8, V8_sse::load_aligned( src ) * V8_sse( 2.f ) );
    }

    // a width that is NOT a power of two: the split is `prev_pow_2( 5 ) = 4` plus a remainder of 1
    {
        using V5 = asimd::SimdVec<float,5,Sse>;
        CHECK( V5::size() == 5 );
        const float expected[ 5 ] = { 2, 3, 4, 5, 6 };
        CHECK_LANES( expected, 5, V5::load_unaligned( src ) + V5( 1.f ) );
    }

    return report( "test_split" );
}
