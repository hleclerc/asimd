// x86 COVERAGE, value by value, ACROSS THE WIDTHS AND THE TYPES -- not just `float x 8`.
//
// `test_ops.cpp` checks every operation once, at the one width the clipping kernel used. This file
// asks the other question: does the SAME operation still hold at 2, 4, 8 and 16 lanes, on FP64 and
// on the integer types, and at the widths that are not a power of two? That is where the dispatch
// picks a different variant -- and where it turns out to pick a wrong one.
//
// There are no `KNOWN_BROKEN` entries left in this file: every bug the audit diagnosed here has
// been fixed, and each one turned into a loud XPASS on the way -- which is the mechanism working
// as intended. See FINDINGS.md for the causes. (`check.h` still carries `KNOWN_BROKEN`, and the
// test harnesses now print `XPASS`, which they had been filtering out.)

#include <asimd/SimdOpsPlus.h>

#include "check.h"

using namespace asimd;

// An architecture capped at SSE2, to exercise the SPLIT path on a machine that has far more: the
// result must not depend on the register width the split happens to land on.
using Sse = X86Cpu<64, features::SSE2, features::SSE>;


// =============================================================================================
// THE GRID. The same checks, driven over (type, width) -- because the registrations under test
// now number in the hundreds, and a value check at one width says nothing about the next one.
//
// Each of these lands on a different variant depending on the target: `gt` on `float x 8` is
// `vcmpps` under AVX, a `k` register under AVX-512VL, and a lane loop under SSE2. All three must
// give the same answer, which is exactly what a grid test is for.
// =============================================================================================
template<class T,int N>
static void grid( const char *label ) {
    using V = SimdVec<T,N>;
    using I = SimdVec<SI32,N>;

    alignas( 64 ) T sa[ 64 ], sb[ 64 ];
    for ( int i = 0; i < N; ++i ) { sa[ i ] = T( i ); sb[ i ] = T( N / 2 ); }
    const V a = V::load_aligned( sa ), b = V::load_aligned( sb );

    // ---- the four comparisons, materialised as bits
    PI64 e_gt = 0, e_lt = 0, e_eq = 0, e_ge = 0;
    for ( int i = 0; i < N; ++i ) {
        if ( sa[ i ] >  sb[ i ] ) e_gt |= PI64( 1 ) << i;
        if ( sa[ i ] <  sb[ i ] ) e_lt |= PI64( 1 ) << i;
        if ( sa[ i ] == sb[ i ] ) e_eq |= PI64( 1 ) << i;
        if ( sa[ i ] >= sb[ i ] ) e_ge |= PI64( 1 ) << i;
    }
    CHECK_AT( label, to_bits( asimd::gt( a, b ) ) == e_gt );
    CHECK_AT( label, to_bits( asimd::lt( a, b ) ) == e_lt );
    CHECK_AT( label, to_bits( asimd::eq( a, b ) ) == e_eq );
    CHECK_AT( label, to_bits( asimd::ge( a, b ) ) == e_ge );

    // ---- the lazy comparison operators go through their own path
    CHECK_AT( label, to_bits( a > b ) == e_gt );
    CHECK_AT( label, to_bits( a < b ) == e_lt );

    // ---- ... and `any`/`all` ON A LAZY COMPARISON go through a THIRD one: `as_a_simd_mask`,
    // where `to_bits( a > b )` and `select( a > b, ... )` both route through `ops::cmp_gt` and
    // never touch it. Which is how its split branch stayed broken -- it assumed both halves
    // returned the BIT flavour of mask, so at 16 lanes with a register form at 8 (i.e. `-mavx`
    // and up) the cell was a hard COMPILE error, not a slow path. Found on the ARM port, where
    // the register stops at 128 bits and the same shape occurs at an ordinary width.
    CHECK_AT( label, any( a > b ) == ( e_gt != 0 ) );
    CHECK_AT( label, all( a > b ) == ( e_gt == ( N >= 64 ? ~PI64( 0 ) : ( PI64( 1 ) << N ) - 1 ) ) );

    // ---- select, driven by both a computed mask and a comparison
    {
        double e[ 64 ];
        for ( int i = 0; i < N; ++i ) e[ i ] = double( ( i & 1 ) ? sa[ i ] : sb[ i ] );
        PI64 alt = 0;
        for ( int i = 0; i < N; ++i ) if ( i & 1 ) alt |= PI64( 1 ) << i;
        CHECK_AT( label, lanes_are( select( mask_from_bits<N>( alt ), a, b ), e, N ) );

        for ( int i = 0; i < N; ++i ) e[ i ] = double( sa[ i ] > sb[ i ] ? sa[ i ] : sb[ i ] );
        CHECK_AT( label, lanes_are( select( a > b, a, b ), e, N ) );
    }

    // ---- to_bits / mask_from_bits round trip, including the top lane
    {
        const PI64 all_set = N >= 64 ? ~PI64( 0 ) : ( PI64( 1 ) << N ) - 1;
        CHECK_AT( label, to_bits( mask_from_bits<N>( all_set ) ) == all_set );
        CHECK_AT( label, to_bits( mask_from_bits<N>( PI64( 1 ) << ( N - 1 ) ) ) == PI64( 1 ) << ( N - 1 ) );
        CHECK_AT( label, all( mask_from_bits<N>( all_set ) ) == true );
        CHECK_AT( label, any( mask_from_bits<N>( 0 ) ) == false );
    }

    // ---- permute: reverse the lanes, which no shuffle immediate could express
    {
        alignas( 64 ) SI32 rev[ 64 ];
        double e[ 64 ];
        for ( int i = 0; i < N; ++i ) { rev[ i ] = N - 1 - i; e[ i ] = double( sa[ N - 1 - i ] ); }
        CHECK_AT( label, lanes_are( permute( a, I::load_aligned( rev ) ), e, N ) );
    }

    // ---- bcast_lane, both the lane-0 special case and a general one
    {
        double e0[ 64 ], e1[ 64 ];
        for ( int i = 0; i < N; ++i ) { e0[ i ] = double( sa[ 0 ] ); e1[ i ] = double( sa[ 1 ] ); }
        CHECK_AT( label, lanes_are( bcast_lane<0>( a ), e0, N ) );
        CHECK_AT( label, lanes_are( bcast_lane<1>( a ), e1, N ) );
    }

    // ---- fma, and the arithmetic operators
    {
        double e[ 64 ];
        for ( int i = 0; i < N; ++i ) e[ i ] = double( T( sa[ i ] * sb[ i ] + sb[ i ] ) );
        CHECK_AT( label, lanes_are( fma( a, b, b ), e, N ) );
        for ( int i = 0; i < N; ++i ) e[ i ] = double( T( sa[ i ] + sb[ i ] ) );
        CHECK_AT( label, lanes_are( a + b, e, N ) );
        for ( int i = 0; i < N; ++i ) e[ i ] = double( T( sa[ i ] * sb[ i ] ) );
        CHECK_AT( label, lanes_are( a * b, e, N ) );
        for ( int i = 0; i < N; ++i ) e[ i ] = double( T( sa[ i ] - sb[ i ] ) );
        CHECK_AT( label, lanes_are( a - b, e, N ) );
    }

    // ---- min / max, and the horizontal sum
    {
        double e[ 64 ];
        for ( int i = 0; i < N; ++i ) e[ i ] = double( sa[ i ] < sb[ i ] ? sa[ i ] : sb[ i ] );
        CHECK_AT( label, lanes_are( min( a, b ), e, N ) );
        for ( int i = 0; i < N; ++i ) e[ i ] = double( sa[ i ] > sb[ i ] ? sa[ i ] : sb[ i ] );
        CHECK_AT( label, lanes_are( max( a, b ), e, N ) );

        T sum = T( 0 );
        for ( int i = 0; i < N; ++i ) sum = T( sum + sa[ i ] );
        CHECK_AT( label, a.sum() == sum );
    }

    // ---- iota, plain and STRIDED. The strided form is the one that had no cell anywhere: it
    // used to be a compile error at a register-backed width (it recursed into a `split` a
    // register impl does not have -- FINDINGS.md finding 1), and after that was fixed it was
    // still a compile error on the 8- and 16-bit lane types at any width that splits, because
    // `beg + n * mul` integer-promotes and the recursion could no longer deduce `T`. Both are
    // fixed; this is what says so.
    {
        double e[ 64 ];
        for ( int i = 0; i < N; ++i ) e[ i ] = double( T( T( 3 ) + T( i ) ) );
        CHECK_AT( label, lanes_are( V::iota( T( 3 ) ), e, N ) );
        for ( int i = 0; i < N; ++i ) e[ i ] = double( T( T( 3 ) + T( i ) * T( 2 ) ) );
        CHECK_AT( label, lanes_are( V::iota( T( 3 ), T( 2 ) ), e, N ) );
    }
}

int main() {
    // the grid first: every (type, width) the x86 backend claims, plus the widths that split.
    grid<FP32, 4>( "FP32x4"  ); grid<FP32, 8>( "FP32x8"  ); grid<FP32,16>( "FP32x16" );
    grid<FP64, 2>( "FP64x2"  ); grid<FP64, 4>( "FP64x4"  ); grid<FP64, 8>( "FP64x8"  );
    grid<SI32, 4>( "SI32x4"  ); grid<SI32, 8>( "SI32x8"  ); grid<SI32,16>( "SI32x16" );
    grid<PI32, 4>( "PI32x4"  ); grid<PI32, 8>( "PI32x8"  ); grid<PI32,16>( "PI32x16" );
    grid<SI64, 2>( "SI64x2"  ); grid<SI64, 4>( "SI64x4"  ); grid<SI64, 8>( "SI64x8"  );
    grid<PI64, 2>( "PI64x2"  ); grid<PI64, 4>( "PI64x4"  ); grid<PI64, 8>( "PI64x8"  );
    grid<SI16,16>( "SI16x16" ); grid<SI16,32>( "SI16x32" );
    grid<FP32, 5>( "FP32x5"  ); grid<FP32, 3>( "FP32x3"  ); // widths that are not a power of two
    grid<FP32,32>( "FP32x32" );                             // wider than any register

    // =========================================================================================
    // 1. load / store, every width, aligned and unaligned
    // =========================================================================================
    {
        alignas( 64 ) float s[ 32 ];                     // 32, because a SimdVec<FP32,32> below
        for ( int i = 0; i < 32; ++i ) s[ i ] = float( i ); // reads 32 lanes out of it
        double e[ 32 ];
        for ( int i = 0; i < 32; ++i ) e[ i ] = i;

        CHECK_LANES( e, 4,  SimdVec<FP32, 4>::load_aligned  ( s ) );
        CHECK_LANES( e, 8,  SimdVec<FP32, 8>::load_aligned  ( s ) );
        CHECK_LANES( e, 16, SimdVec<FP32,16>::load_aligned  ( s ) );
        CHECK_LANES( e, 8,  SimdVec<FP32, 8>::load_unaligned( s ) );
        CHECK_LANES( e, 32, SimdVec<FP32,32>::load_unaligned( s ) ); // wider than any register
        // the same eight lanes, forced through two SSE2 registers
        CHECK_LANES( e, 8,  SimdVec<FP32, 8, Sse>::load_aligned( s ) );
        // widths that are not a power of two: prev_pow_2 splits 5 into 4 + 1, 3 into 2 + 1
        CHECK_LANES( e, 5,  SimdVec<FP32, 5>::load_unaligned( s ) );
        CHECK_LANES( e, 3,  SimdVec<FP32, 3>::load_unaligned( s ) );

        alignas( 64 ) double d[ 8 ];
        for ( int i = 0; i < 8; ++i ) d[ i ] = i;
        CHECK_LANES( e, 2, SimdVec<FP64,2>::load_aligned( d ) );
        CHECK_LANES( e, 4, SimdVec<FP64,4>::load_aligned( d ) );
        CHECK_LANES( e, 8, SimdVec<FP64,8>::load_aligned( d ) );

        // a round trip through store, at a width that splits
        alignas( 64 ) float out[ 8 ] = {};
        SimdVec<FP32,5>::load_unaligned( s ).store_unaligned( out );
        CHECK( out[ 0 ] == 0 && out[ 4 ] == 4 );
    }

    // =========================================================================================
    // 2. arithmetic, every width and every type that has a register form
    // =========================================================================================
    {
        const double e[ 16 ] = { 5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5 };
        CHECK_LANES( e, 4,  SimdVec<FP32, 4>( 2.f ) + SimdVec<FP32, 4>( 3.f ) );
        CHECK_LANES( e, 8,  SimdVec<FP32, 8>( 2.f ) + SimdVec<FP32, 8>( 3.f ) );
        CHECK_LANES( e, 16, SimdVec<FP32,16>( 2.f ) + SimdVec<FP32,16>( 3.f ) );
        CHECK_LANES( e, 2,  SimdVec<FP64, 2>( 2.  ) + SimdVec<FP64, 2>( 3.  ) );
        CHECK_LANES( e, 4,  SimdVec<FP64, 4>( 2.  ) + SimdVec<FP64, 4>( 3.  ) );
        CHECK_LANES( e, 8,  SimdVec<FP64, 8>( 2.  ) + SimdVec<FP64, 8>( 3.  ) );
        CHECK_LANES( e, 4,  SimdVec<SI32, 4>( 2   ) + SimdVec<SI32, 4>( 3   ) );
        CHECK_LANES( e, 8,  SimdVec<SI32, 8>( 2   ) + SimdVec<SI32, 8>( 3   ) );
        CHECK_LANES( e, 4,  SimdVec<SI64, 4>( 2   ) + SimdVec<SI64, 4>( 3   ) );
        // and through the split
        CHECK_LANES( e, 8,  SimdVec<FP32,8,Sse>( 2.f ) + SimdVec<FP32,8,Sse>( 3.f ) );
        CHECK_LANES( e, 5,  SimdVec<FP32,5    >( 2.f ) + SimdVec<FP32,5    >( 3.f ) );
    }
    {
        const double e[ 16 ] = { 6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6 };
        CHECK_LANES( e, 4,  SimdVec<FP32, 4>( 2.f ) * SimdVec<FP32, 4>( 3.f ) );
        CHECK_LANES( e, 16, SimdVec<FP32,16>( 2.f ) * SimdVec<FP32,16>( 3.f ) );
        CHECK_LANES( e, 8,  SimdVec<FP64, 8>( 2.  ) * SimdVec<FP64, 8>( 3.  ) );
        CHECK_LANES( e, 5,  SimdVec<FP32, 5>( 2.f ) * SimdVec<FP32, 5>( 3.f ) );
    }
    {   // min / max on signed integers
        alignas( 64 ) SI32 a[ 8 ] = { -5, -4, -3, -2, -1, 0, 1, 2 };
        const double lo[ 8 ] = { -5, -4, -3, -2, -1, 0, 0, 0 };
        const double hi[ 8 ] = {  0,  0,  0,  0,  0, 0, 1, 2 };
        CHECK_LANES( lo, 8, min( SimdVec<SI32,8>::load_aligned( a ), SimdVec<SI32,8>( 0 ) ) );
        CHECK_LANES( hi, 8, max( SimdVec<SI32,8>::load_aligned( a ), SimdVec<SI32,8>( 0 ) ) );
    }

    // =========================================================================================
    // 3. fma, permute, bcast_lane -- at the widths `test_ops.cpp` does not reach
    // =========================================================================================
    {
        const double e[ 16 ] = { 7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7 };
        CHECK_LANES( e, 4,  fma( SimdVec<FP32, 4>( 2.f ), SimdVec<FP32, 4>( 3.f ), SimdVec<FP32, 4>( 1.f ) ) );
        CHECK_LANES( e, 16, fma( SimdVec<FP32,16>( 2.f ), SimdVec<FP32,16>( 3.f ), SimdVec<FP32,16>( 1.f ) ) );
        CHECK_LANES( e, 4,  fma( SimdVec<FP64, 4>( 2.  ), SimdVec<FP64, 4>( 3.  ), SimdVec<FP64, 4>( 1.  ) ) );
        CHECK_LANES( e, 8,  fma( SimdVec<FP64, 8>( 2.  ), SimdVec<FP64, 8>( 3.  ), SimdVec<FP64, 8>( 1.  ) ) );
        CHECK_LANES( e, 8,  fma( SimdVec<FP32,8,Sse>( 2.f ), SimdVec<FP32,8,Sse>( 3.f ), SimdVec<FP32,8,Sse>( 1.f ) ) );
    }
    {   // permute, variable indices, reversing
        alignas( 64 ) float s16[ 16 ]; for ( int i = 0; i < 16; ++i ) s16[ i ] = float( i );
        alignas( 64 ) SI32  r16[ 16 ]; for ( int i = 0; i < 16; ++i ) r16[ i ] = 15 - i;
        double e16[ 16 ];              for ( int i = 0; i < 16; ++i ) e16[ i ] = 15 - i;
        CHECK_LANES( e16, 16, permute( SimdVec<FP32,16>::load_aligned( s16 ),
                                       SimdVec<SI32,16>::load_aligned( r16 ) ) );

        alignas( 64 ) float s4[ 4 ] = { 0, 1, 2, 3 };
        alignas( 64 ) SI32  r4[ 4 ] = { 3, 2, 1, 0 };
        const double e4[ 4 ] = { 3, 2, 1, 0 };
        CHECK_LANES( e4, 4, permute( SimdVec<FP32,4>::load_aligned( s4 ),
                                     SimdVec<SI32,4>::load_aligned( r4 ) ) );

        // A width that is NOT a power of two. The generic `permute` used to mask the index with
        // `& ( N - 1 )`, which is a modulo only at a power of two -- and arbitrary widths are
        // what asimd is for. At N = 5 the mask was `& 4` and this returned `50 10 10 10 10`.
        alignas( 64 ) float s5[ 5 ] = { 10, 20, 30, 40, 50 };
        alignas( 64 ) SI32  r5[ 5 ] = { 4, 3, 2, 1, 0 };
        const double e5[ 5 ] = { 50, 40, 30, 20, 10 };
        CHECK_LANES( e5, 5, permute( SimdVec<FP32,5>::load_unaligned( s5 ),
                                     SimdVec<SI32,5>::load_unaligned( r5 ) ) );
    }
    {   // bcast_lane
        alignas( 64 ) double s[ 4 ] = { 1, 2, 3, 4 };
        const double e[ 4 ] = { 3, 3, 3, 3 };
        CHECK_LANES( e, 4, bcast_lane<2>( SimdVec<FP64,4>::load_aligned( s ) ) );
    }

    // =========================================================================================
    // 4. masks: to_bits, mask_from_bits, select, all / any
    // =========================================================================================
    {
        alignas( 64 ) float a8[ 8 ] = { -1, 2, -3, 4, 5, -6, 7, 8 };
        CHECK( to_bits( SimdVec<FP32,8>::load_aligned( a8 ) > SimdVec<FP32,8>( 0.f ) ) == 0xdau );

        alignas( 64 ) float a16[ 16 ] = { -1,2,-3,4,5,-6,7,8, -1,2,-3,4,5,-6,7,8 };
        CHECK( to_bits( SimdVec<FP32,16>::load_aligned( a16 ) > SimdVec<FP32,16>( 0.f ) ) == 0xdadau );

        alignas( 64 ) double a4[ 4 ] = { -1, 2, -3, 4 };
        CHECK( to_bits( SimdVec<FP64,4>::load_aligned( a4 ) > SimdVec<FP64,4>( 0. ) ) == 0xau );

        // round trip through both flavours of mask
        CHECK( to_bits( mask_from_bits< 4>( 0x5u    ) ) == 0x5u    );
        CHECK( to_bits( mask_from_bits< 8>( 0x5au   ) ) == 0x5au   );
        CHECK( to_bits( mask_from_bits<16>( 0x1234u ) ) == 0x1234u );

        // integer comparisons, including on negative values
        using I = SimdVec<SI32,8>;
        alignas( 64 ) SI32 n[ 8 ] = { -5, -4, -3, -2, -1, 0, 1, 2 };
        CHECK( to_bits( eq( I::iota( 0 ), I( 3 ) ) )         == 0x08u );
        CHECK( to_bits( ge( I::iota( 0 ), I( 5 ) ) )         == 0xe0u );
        CHECK( to_bits( ge( I::load_aligned( n ), I( 0 ) ) ) == 0xe0u );

        // select, both mask flavours, several widths
        {   alignas( 64 ) double s[ 4 ] = { 1, 2, 3, 4 };
            const double e[ 4 ] = { 9, 2, 9, 4 };
            CHECK_LANES( e, 4, select( mask_from_bits<4>( 0x5u ), SimdVec<FP64,4>( 9. ),
                                       SimdVec<FP64,4>::load_aligned( s ) ) ); }
        {   double e[ 16 ]; for ( int i = 0; i < 16; ++i ) e[ i ] = ( i & 1 ) ? 0 : 1;
            CHECK_LANES( e, 16, select( mask_from_bits<16>( 0x5555u ), SimdVec<FP32,16>( 1.f ),
                                        SimdVec<FP32,16>( 0.f ) ) ); }

        CHECK( any( mask_from_bits<8>( 0x01u ) ) == true  );
        CHECK( any( mask_from_bits<8>( 0x00u ) ) == false );
        CHECK( all( mask_from_bits<4>( 0xfu  ) ) == true  );

        // `all` on an AVX-512 mask register used to be `~mask.data.reg == 0`, where `~` integer-
        // promotes __mmask8 to int: `~0xff` is 0xffffff00, never zero, so `all` was ALWAYS false
        // at every width but 64. It must complement inside the mask type.
        CHECK( all( mask_from_bits< 8>( 0xffu   ) ) == true  );
        CHECK( all( mask_from_bits< 8>( 0xfeu   ) ) == false );
        CHECK( all( mask_from_bits<16>( 0xffffu ) ) == true  );
        CHECK( all( mask_from_bits<16>( 0xfffeu ) ) == false );
    }

    // =========================================================================================
    // 5. UNSIGNED types must not be routed through SIGNED instructions
    // =========================================================================================
    {
        using U = SimdVec<PI32,8>;
        alignas( 64 ) PI32 a[ 8 ] = { 1, 2, 3, 4, 5, 6, 7, 0xFFFFFFFFu };
        const double lo[ 8 ] = { 1, 2, 3, 4, 5, 6, 7, 9 };
        const double hi[ 8 ] = { 9, 9, 9, 9, 9, 9, 9, 4294967295.0 };
        // These used to go to `_mm*_min_epi32` -- the SIGNED instruction -- so 0xFFFFFFFF was
        // read as -1 and `min` returned it.
        CHECK_LANES( lo, 8, min( U::load_aligned( a ), U( 9 ) ) );
        CHECK_LANES( hi, 8, max( U::load_aligned( a ), U( 9 ) ) );

        using U64 = SimdVec<PI64,4>;
        alignas( 64 ) PI64 a64[ 4 ] = { 1, 2, 3, 0xFFFFFFFFFFFFFFFFull };
        const double lo64[ 4 ] = { 1, 2, 3, 9 };
        CHECK_LANES( lo64, 4, min( U64::load_aligned( a64 ), U64( 9 ) ) );

        // and the unsigned COMPARISON, which has no instruction at all before AVX-512: it is the
        // signed compare on operands with their sign bit flipped.
        alignas( 64 ) PI32 c[ 8 ] = { 0xFFFFFFFFu, 1, 1, 1, 1, 1, 1, 1 };
        CHECK( to_bits( gt( U::load_aligned( c ), U( 1 ) ) ) == 0x01u );
        alignas( 64 ) SI32 d[ 8 ] = { -1, 1, 1, 1, 1, 1, 1, 1 };
        CHECK( to_bits( gt( SimdVec<SI32,8>::load_aligned( d ), SimdVec<SI32,8>( 1 ) ) ) == 0x00u );
    }

    // =========================================================================================
    // 6. iota, gather
    // =========================================================================================
    {
        const double e[ 16 ] = { 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18 };
        CHECK_LANES( e, 8,  SimdVec<SI32, 8>::iota( 3 ) );
        CHECK_LANES( e, 16, SimdVec<SI32,16>::iota( 3 ) );
        CHECK_LANES( e, 8,  SimdVec<FP32, 8>::iota( 3 ) );
        CHECK_LANES( e, 5,  SimdVec<FP32, 5>::iota( 3 ) );
        // `iota( beg, mul )` IS tested now -- in the grid above, at every (type, width). It used
        // to be impossible to call at a register-backed width at all, and the grid is where the
        // remaining half of the problem showed up: the narrow lane types.
        {   const double e2[ 8 ] = { 3, 5, 7, 9, 11, 13, 15, 17 };
            CHECK_LANES( e2, 8, SimdVec<SI32,8>::iota( 3, 2 ) ); }
    }
    {
        alignas( 64 ) float data[ 16 ]; for ( int i = 0; i < 16; ++i ) data[ i ] = float( 10 * i );
        alignas( 64 ) SI32  idx [  8 ] = { 7, 6, 5, 4, 3, 2, 1, 0 };
        const double e[ 8 ] = { 70, 60, 50, 40, 30, 20, 10, 0 };
        CHECK_LANES( e, 8, SimdVec<FP32,8>::gather( data, SimdVec<SI32,8>::load_aligned( idx ) ) );
    }

    return report( "test_x86_ops" );
}
