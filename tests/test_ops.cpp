// Every operation added by `SimdOpsPlus.h`, value by value.
#include <asimd/SimdOpsPlus.h>

#include "check.h"

using V = asimd::SimdVec<float,8>;
using I = asimd::SimdVec<asimd::SI32,8>;

int main() {
    alignas( 64 ) float src[ 8 ] = { -1, 2, -3, 4, 5, -6, 7, 8 };
    const V v = V::load_aligned( src );

    // ---- to_bits: the sign mask as an integer. This is what carries the `ctz` and the
    // rotations, hence everything that replaces a traversal in a branchless clip.
    const unsigned bits = asimd::to_bits( v > V( 0.f ) );
    CHECK( bits == 0xda );                               // 0,1,1,0,1,1,0,1 -> 0b11011010

    // ---- mask_from_bits: the dual. Must give back exactly what `to_bits` read.
    {
        const float expected[ 8 ] = { 0, 1, 0, 1, 1, 0, 1, 1 };
        CHECK_LANES( expected, 8, asimd::select( asimd::mask_from_bits<8>( bits ), V( 1.f ), V( 0.f ) ) );
    }

    // ---- permute, with VARIABLE indices
    {
        alignas( 64 ) asimd::SI32 idx[ 8 ] = { 7, 6, 5, 4, 3, 2, 1, 0 };
        const float expected[ 8 ] = { 8, 7, -6, 5, 4, -3, 2, -1 };
        CHECK_LANES( expected, 8, asimd::permute( v, I::load_aligned( idx ) ) );
    }

    // ---- select, driven by a lazy comparison
    {
        const float expected[ 8 ] = { -1, 9, -3, 9, 9, -6, 9, 9 };
        CHECK_LANES( expected, 8, asimd::select( v > V( 0.f ), V( 9.f ), v ) );
    }

    // ---- fma, and the operators that were missing
    {
        const float expected[ 8 ] = { -1, 5, -5, 9, 11, -11, 15, 17 };
        CHECK_LANES( expected, 8, asimd::fma( v, V( 2.f ), V( 1.f ) ) );
    }
    {
        const float expected[ 8 ] = { -4, 5, -10, 11, 14, -19, 20, 23 };
        CHECK_LANES( expected, 8, v * V( 3.f ) - V( 1.f ) );
    }
    {
        const float expected[ 8 ] = { -0.5f, 1, -1.5f, 2, 2.5f, -3, 3.5f, 4 };
        CHECK_LANES( expected, 8, v / V( 2.f ) );
    }

    // ---- eq / ge, and "lane i"
    {
        const I io = I::iota( 0 );
        const asimd::SI32 e1[ 8 ] = { 0, 1, 2, 99, 4, 5, 6, 7 };
        CHECK_LANES( e1, 8, asimd::select( asimd::eq( io, I( 3 ) ), I( 99 ), io ) );
        const asimd::SI32 e2[ 8 ] = { 0, 1, 2, 3, 4, 0, 0, 0 };
        CHECK_LANES( e2, 8, asimd::select( asimd::ge( io, I( 5 ) ), I( 0 ), io ) );
    }

    // ---- bcast_lane, compile-time lane
    {
        const float expected[ 8 ] = { 2, 2, 2, 2, 2, 2, 2, 2 };
        CHECK_LANES( expected, 8, asimd::bcast_lane<1>( v ) );
    }

    // ---- iota has a register form; without it gcc builds it with successive `vpinsrd`
    {
        const asimd::SI32 expected[ 8 ] = { 3, 4, 5, 6, 7, 8, 9, 10 };
        CHECK_LANES( expected, 8, I::iota( 3 ) );
    }

    return report( "test_ops" );
}
