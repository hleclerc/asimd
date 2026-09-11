// ARM COVERAGE, value by value, ACROSS THE WIDTHS, THE TYPES AND THE TWO FEATURE LEVELS.
//
// The counterpart of `test_x86_ops.cpp`. The grid is deliberately the same shape -- an operation
// that holds at four lanes says nothing about eight -- with three additions that are specific to
// this architecture:
//
//   THE SAME GRID AT THE ARMv7-A FLOOR. `ArmCpu<64,NEON,FMA>` is NEON without ASIMD: no double
//   lanes, no `fdiv`, no `ADDV`, no `TBL`, no 64-bit compares. Every one of those has to
//   disappear and leave a CORRECT answer behind, and the answer has to be the same one the A64
//   path gives. This is the only way to exercise the 32-bit ARM lattice without a 32-bit ARM
//   part, and it is what says the two feature levels are really separate -- registering an A64
//   instruction against `NEON` would compile fine here and not at all on the target that needs
//   it, which is precisely the failure the x86 lattice was fixed for.
//
//   THE SPLIT AT EVERY WIDTH ABOVE FOUR. On x86 that is an edge case; here the register is 128
//   bits and never wider, so `SimdVec<float,8>` -- the README's opening example -- IS the split
//   path. It gets the same attention the register widths do.
//
//   THE OPERATIONS ARM HAS AND x86 DOES NOT. A per-lane variable shift, an integer
//   multiply-accumulate, native unsigned compares. Those have no cell in the x86 tests, so they
//   need one here or they are registered and never checked.

#include <asimd/asimd.h>

#include "check.h"

using namespace asimd;

/// NEON without ASIMD: the ARMv7-A floor, forced on whatever machine this is compiled on.
using V7 = ArmCpu<64, features::NEON, features::FMA>;


/// compares `n` lanes of a raw buffer against what they should be, and says WHICH lane differs.
///
/// NOT `bool same = true; ... same &= ( got[ i ] == want[ i ] )`, which is what this file used to
/// do in two places. That shape has two faults and both of them cost a CI round trip:
///
///   IT REPORTS NOTHING. The failure printed `[SI16x16] same` -- true, something differed, on a
///   compiler nobody here can run. Which lane, and by how much, is the whole of the information.
///
///   IT IS A BOOL REDUCTION OVER A COUNTED LOOP, which is exactly the shape an auto-vectorizer
///   reaches for. So it was simultaneously the least informative formulation available and the
///   most likely to be miscompiled -- and it failed on MSVC at `/arch:AVX2` only, where nothing
///   in the library differs from `/arch:AVX` at all: `SimdVec<SI16,16>` has no register impl on
///   x86 below AVX-512BW, so it is the same splittable path, same `split_size_0`, at both levels.
/// THE PREFIX IS `info`, deliberately, and the lesson is worth more than the check. The first
/// version of this printed `  [SI16x16] ...`, and BOTH test harnesses threw it away:
/// `run_all_isa.sh` shows `^  broken|^  FAIL` and `run_msvc.ps1` shows `^$t|FAIL|broken`. So the
/// diagnostic was produced, on the one compiler that needed it, and discarded by the script --
/// costing a full CI round trip to learn nothing. (The same filters were also swallowing
/// `check.h`'s `XPASS`, which is how a fixed known-broken bug would have gone unnoticed.) Both
/// filters pass `info` now, and this reads like the verdicts beside it.
///
/// It dumps BOTH BUFFERS, not just the first bad lane: the shape of a failure is the diagnosis. An
/// untouched upper half points at the split store; scattered differences point at the data.
/// ONE template parameter, not two `auto`s. The two buffers are necessarily the same type, and
/// saying so makes a mismatched call a compile error instead of a silent comparison between
/// different types -- and removes an abbreviated-template-parameter form from a file that has to
/// go through MSVC unseen.
template<class T>
static bool lanes_match( const char *label, const char *what, const T *got, const T *want, int n ) {
    int bad = -1;
    for ( int i = 0; i < n && bad < 0; ++i ) if ( got[ i ] != want[ i ] ) bad = i;
    if ( bad < 0 ) return true;

    printf( "  info   [%s] %s: first bad lane %d of %d\n", label, what, bad, n );
    printf( "  info     got " ); for ( int i = 0; i < n; ++i ) printf( " %g", double( got [ i ] ) );
    printf( "\n  info     want" ); for ( int i = 0; i < n; ++i ) printf( " %g", double( want[ i ] ) );
    printf( "\n" );
    return false;
}

// =============================================================================================
// THE GRID, over (type, width, architecture).
//
// Each cell lands on a different variant: `gt` on `float x 4` is `CMGT` under NEON and
// `float x 8` is two of them under the split rank, while at the ARMv7 floor `double` is a scalar
// loop. All of them must give the same answer.
// =============================================================================================
template<class T,int N,class Arch = NativeCpu>
static void grid( const char *label ) {
    using V = SimdVec<T,N,Arch>;
    using I = SimdVec<SI32,N,Arch>;

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

    // ---- the lazy comparison operators go through their own path (`internal::gt`, not
    // `ops::cmp_gt`): two mechanisms, two sets of registrations, both have to be right.
    CHECK_AT( label, to_bits( a > b ) == e_gt );
    CHECK_AT( label, to_bits( a < b ) == e_lt );
    CHECK_AT( label, any( a > b ) == ( e_gt != 0 ) );
    CHECK_AT( label, all( a > b ) == ( e_gt == ( N >= 64 ? ~PI64( 0 ) : ( PI64( 1 ) << N ) - 1 ) ) );

    // ---- select, driven by both a computed mask and a comparison
    {
        double e[ 64 ];
        for ( int i = 0; i < N; ++i ) e[ i ] = double( ( i & 1 ) ? sa[ i ] : sb[ i ] );
        PI64 alt = 0;
        for ( int i = 0; i < N; ++i ) if ( i & 1 ) alt |= PI64( 1 ) << i;
        CHECK_AT( label, lanes_are( select( mask_from_bits<N,Arch>( alt ), a, b ), e, N ) );

        for ( int i = 0; i < N; ++i ) e[ i ] = double( sa[ i ] > sb[ i ] ? sa[ i ] : sb[ i ] );
        CHECK_AT( label, lanes_are( select( a > b, a, b ), e, N ) );
    }

    // ---- to_bits / mask_from_bits round trip, including the top lane
    {
        const PI64 all_set = N >= 64 ? ~PI64( 0 ) : ( PI64( 1 ) << N ) - 1;
        CHECK_AT( label, to_bits( mask_from_bits<N,Arch>( all_set ) ) == all_set );
        CHECK_AT( label, to_bits( mask_from_bits<N,Arch>( PI64( 1 ) << ( N - 1 ) ) ) == PI64( 1 ) << ( N - 1 ) );
        CHECK_AT( label, all( mask_from_bits<N,Arch>( all_set ) ) == true );
        CHECK_AT( label, any( mask_from_bits<N,Arch>( 0 ) ) == false );
        // one bit in the UPPER half, which is what the split `mask_from_bits` gets wrong if it
        // forgets to shift: `b >> n0` for the second half.
        if ( N >= 2 )
            CHECK_AT( label, to_bits( mask_from_bits<N,Arch>( PI64( 1 ) << ( N / 2 ) ) ) == PI64( 1 ) << ( N / 2 ) );
    }

    // ---- permute: reverse the lanes, which no shuffle immediate could express
    {
        alignas( 64 ) SI32 rev[ 64 ];
        double e[ 64 ];
        for ( int i = 0; i < N; ++i ) { rev[ i ] = N - 1 - i; e[ i ] = double( sa[ N - 1 - i ] ); }
        CHECK_AT( label, lanes_are( permute( a, I::load_aligned( rev ) ), e, N ) );
    }

    // ---- bcast_lane. Lane 0, lane 1, and -- the case the split form gets wrong if it forgets
    // to rebase the index -- a lane in the UPPER half.
    {
        double e0[ 64 ], e1[ 64 ], eh[ 64 ];
        for ( int i = 0; i < N; ++i ) {
            e0[ i ] = double( sa[ 0 ] ); e1[ i ] = double( sa[ 1 ] ); eh[ i ] = double( sa[ N - 1 ] );
        }
        CHECK_AT( label, lanes_are( bcast_lane<0>( a ), e0, N ) );
        CHECK_AT( label, lanes_are( bcast_lane<1>( a ), e1, N ) );
        CHECK_AT( label, lanes_are( bcast_lane<N-1>( a ), eh, N ) );
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

    // ---- min / max, and the horizontal sum -- `ADDV` at a register width, a ladder above it
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

    // ---- iota, and a store round trip
    {
        double e[ 64 ];
        for ( int i = 0; i < N; ++i ) e[ i ] = double( T( T( 3 ) + T( i ) ) );
        CHECK_AT( label, lanes_are( V::iota( T( 3 ) ), e, N ) );

        // THE STRIDED FORM, which did not compile on the narrow types at any width that splits:
        // `beg + n * mul` integer-PROMOTES on an 8- or 16-bit lane type, so the recursive call
        // deduced `T` as `int` and as `short` at once. See the note in `iota` in
        // SimdVecImpl_Generic.h.
        for ( int i = 0; i < N; ++i ) e[ i ] = double( T( T( 3 ) + T( i ) * T( 2 ) ) );
        CHECK_AT( label, lanes_are( V::iota( T( 3 ), T( 2 ) ), e, N ) );

        // BOTH SPELLINGS OF THE STORE, separately. `lanes_are` -- and therefore every other
        // check above -- goes through the STATIC two-argument form; the member form is reached by
        // nothing else in the suite, and forwards to it. Checking them apart is what says whether
        // a failure is in the data or in the forwarding.
        // NO `alignas( 64 )` ON THESE TWO, and that is the point of the store being *unaligned*.
        // They had it, for no reason: `store_unaligned` asks nothing of the address. What that
        // bought was two more 64-byte-aligned arrays in a nested block of a function that already
        // holds several, while `sa` and `sb` are still live in the enclosing scope -- a stack
        // layout exotic enough that it is worth not asking for when nothing needs it, on a
        // compiler nobody here can run. `lanes_are` aligns its own buffer, but it lives alone in
        // its own function.
        T o_static[ 64 ] = {};
        T o_member[ 64 ] = {};
        V::store_unaligned( o_static, a );
        a.store_unaligned( o_member );
        CHECK_AT( label, lanes_match( label, "store_unaligned, static form", o_static, sa, N ) );
        CHECK_AT( label, lanes_match( label, "store_unaligned, member form", o_member, sa, N ) );
    }
}

/// the integer-only operations, which the shared grid cannot carry: `<<` on a float is meant to
/// be rejected, not supported.
template<class T,int N,class Arch = NativeCpu>
static void grid_int( const char *label ) {
    using V = SimdVec<T,N,Arch>;

    alignas( 64 ) T sa[ 64 ], sh[ 64 ];
    for ( int i = 0; i < N; ++i ) { sa[ i ] = T( i + 1 ); sh[ i ] = T( i % 3 ); }
    const V a = V::load_aligned( sa ), s = V::load_aligned( sh );

    // A PER-LANE VARIABLE SHIFT, each lane by its own amount. `vshlq_*` is one instruction at
    // every width on every ARM part; x86 has nothing equivalent below AVX2 (`vpsllvd`) and
    // nothing at all on the 8- and 16-bit types. Note the shift amounts DIFFER between lanes --
    // a uniform shift would pass even with a broken register form.
    double e[ 64 ];
    for ( int i = 0; i < N; ++i ) e[ i ] = double( T( sa[ i ] << sh[ i ] ) );
    CHECK_AT( label, lanes_are( a << s, e, N ) );

    // the bitwise `and`, which on ARM is typed integer-only, hence a reinterpret sandwich on the
    // floating point types -- checked here on the integer ones, where it is direct.
    for ( int i = 0; i < N; ++i ) e[ i ] = double( T( sa[ i ] & T( 6 ) ) );
    CHECK_AT( label, lanes_are( a & V( T( 6 ) ), e, N ) );
}

int main() {
    // =========================================================================================
    // 1. the grid, on this target
    // =========================================================================================
    grid<FP32, 4>( "FP32x4"  ); grid<FP32, 8>( "FP32x8"  ); grid<FP32,16>( "FP32x16" );
    grid<FP64, 2>( "FP64x2"  ); grid<FP64, 4>( "FP64x4"  ); grid<FP64, 8>( "FP64x8"  );
    grid<SI32, 4>( "SI32x4"  ); grid<SI32, 8>( "SI32x8"  ); grid<SI32,16>( "SI32x16" );
    grid<PI32, 4>( "PI32x4"  ); grid<PI32, 8>( "PI32x8"  );
    grid<SI64, 2>( "SI64x2"  ); grid<SI64, 4>( "SI64x4"  );
    grid<PI64, 2>( "PI64x2"  ); grid<PI64, 4>( "PI64x4"  );
    grid<SI16, 8>( "SI16x8"  ); grid<SI16,16>( "SI16x16" );
    grid<PI16, 8>( "PI16x8"  );
    grid<SI8 ,16>( "SI8x16"  ); grid<PI8 ,16>( "PI8x16"  );
    grid<FP32, 5>( "FP32x5"  ); grid<FP32, 3>( "FP32x3"  ); // widths that are not a power of two
    grid<FP32,32>( "FP32x32" );                             // eight registers for one value

    grid_int<SI32, 4>( "SI32x4 int"  ); grid_int<SI32, 8>( "SI32x8 int"  );
    grid_int<PI32, 4>( "PI32x4 int"  );
    grid_int<SI64, 2>( "SI64x2 int"  ); grid_int<PI64, 2>( "PI64x2 int"  );
    grid_int<SI16, 8>( "SI16x8 int"  ); grid_int<PI8 ,16>( "PI8x16 int"  );

    // =========================================================================================
    // 2. THE SAME GRID AT THE ARMv7-A FLOOR
    //
    // `double` is absent from `features::NEON`'s type list, so `SimdVec<double,2,V7>` is two
    // scalar lanes rather than a `float64x2_t` -- and it still has to give the same answers.
    // Same for `permute` (no `TBL`), `bcast_lane` (no `DUP` from a q register), `.sum()` (no
    // `ADDV`) and the 64-bit comparisons.
    // =========================================================================================
    grid<FP32, 4,V7>( "v7 FP32x4"  ); grid<FP32, 8,V7>( "v7 FP32x8"  );
    grid<FP32,16,V7>( "v7 FP32x16" ); grid<FP32, 5,V7>( "v7 FP32x5"  );
    grid<FP64, 2,V7>( "v7 FP64x2"  ); grid<FP64, 4,V7>( "v7 FP64x4"  );
    grid<SI32, 4,V7>( "v7 SI32x4"  ); grid<SI32, 8,V7>( "v7 SI32x8"  );
    grid<PI32, 4,V7>( "v7 PI32x4"  );
    grid<SI64, 2,V7>( "v7 SI64x2"  ); grid<PI64, 2,V7>( "v7 PI64x2"  );
    grid<SI16, 8,V7>( "v7 SI16x8"  ); grid<PI8 ,16,V7>( "v7 PI8x16"  );

    grid_int<SI32,4,V7>( "v7 SI32x4 int" ); grid_int<PI32,4,V7>( "v7 PI32x4 int" );

    // =========================================================================================
    // 3. THE TWO FEATURE LEVELS MUST AGREE, lane for lane
    //
    // The grid above checks each against the expected values independently. This checks them
    // against EACH OTHER on a computation with no obvious closed form, which is the shape a real
    // kernel has -- and where a register form that is subtly wrong (a byte index off by one in
    // the `TBL` control, say) shows up as a disagreement rather than as an implausible number.
    // =========================================================================================
    {
        alignas( 64 ) float s[ 8 ] = { -1, 2, -3, 4, 5, -6, 7, 8 };
        alignas( 64 ) SI32  x[ 8 ] = { 7, 0, 5, 2, 3, 4, 1, 6 };

        using Va = SimdVec<float,8>;        using Ia = SimdVec<SI32,8>;
        using Vb = SimdVec<float,8,V7>;     using Ib = SimdVec<SI32,8,V7>;

        const Va a = Va::load_aligned( s ); const Ia ia = Ia::load_aligned( x );
        const Vb b = Vb::load_aligned( s ); const Ib ib = Ib::load_aligned( x );

        alignas( 64 ) float ra[ 8 ], rb[ 8 ];

        // a permutation, then an fma, then a selection driven by a comparison -- the three
        // operations whose ARM forms differ most from their fallbacks.
        fma( permute( a, ia ), Va( 3.f ), select( a > Va( 0.f ), a, Va( -1.f ) ) ).store_unaligned( ra );
        fma( permute( b, ib ), Vb( 3.f ), select( b > Vb( 0.f ), b, Vb( -1.f ) ) ).store_unaligned( rb );
        CHECK( lanes_match( "A vs V7", "fma( permute, select ) must agree", ra, rb, 8 ) );

        CHECK( to_bits( asimd::gt( a, Va( 0.f ) ) ) == to_bits( asimd::gt( b, Vb( 0.f ) ) ) );
        CHECK( a.sum() == b.sum() );
        CHECK( ( a > Va( 0.f ) ).size() == 8 );
    }

    // =========================================================================================
    // 4. `to_bits` AT ALL FOUR LANE WIDTHS
    //
    // There is no `movemask` on ARM: each of these is an AND with a power-of-two constant and an
    // `ADDV`. The 16-lane one is the one to watch -- `vaddvq_u8` returns a `uint8_t`, so summing
    // sixteen bits in one reduction would silently drop the top eight. It reduces the halves
    // separately, and this is the check that says so.
    // =========================================================================================
    {
        alignas( 64 ) float f4[ 4 ] = { -1, 2, -3, 4 };
        CHECK( to_bits( SimdVec<FP32,4>::load_aligned( f4 ) > SimdVec<FP32,4>( 0.f ) ) == 0xau );

        alignas( 64 ) double d2[ 2 ] = { -1, 2 };
        CHECK( to_bits( SimdVec<FP64,2>::load_aligned( d2 ) > SimdVec<FP64,2>( 0. ) ) == 0x2u );

        alignas( 64 ) SI16 h8[ 8 ] = { -1, 2, -3, 4, 5, -6, 7, 8 };
        CHECK( to_bits( SimdVec<SI16,8>::load_aligned( h8 ) > SimdVec<SI16,8>( 0 ) ) == 0xdau );

        // SIXTEEN lanes of eight bits, with bits set in BOTH halves and in the top lane.
        alignas( 64 ) SI8 c16[ 16 ] = { -1, 2, -3, 4, 5, -6, 7, 8, -9, 10, -11, 12, 13, -14, 15, 16 };
        CHECK( to_bits( SimdVec<SI8,16>::load_aligned( c16 ) > SimdVec<SI8,16>( 0 ) ) == 0xdadau );

        // and the split widths, where the halves are shifted into place
        alignas( 64 ) float f8[ 8 ] = { -1, 2, -3, 4, 5, -6, 7, 8 };
        CHECK( to_bits( SimdVec<FP32,8>::load_aligned( f8 ) > SimdVec<FP32,8>( 0.f ) ) == 0xdau );
        alignas( 64 ) float f16[ 16 ] = { -1,2,-3,4,5,-6,7,8, -1,2,-3,4,5,-6,7,8 };
        CHECK( to_bits( SimdVec<FP32,16>::load_aligned( f16 ) > SimdVec<FP32,16>( 0.f ) ) == 0xdadau );
    }

    // =========================================================================================
    // 5. UNSIGNED TYPES MUST NOT BE ROUTED THROUGH SIGNED INSTRUCTIONS
    //
    // On ARM the signedness is part of the instruction NAME -- `CMHI` against `CMGT`, `UMIN`
    // against `SMIN` -- which makes this table hard to get wrong. It was got wrong on the x86
    // side, where the unsigned types were routed to `_mm_min_epi32` and `min( 0xFFFFFFFF, 9 )`
    // returned 0xFFFFFFFF. Checked here anyway: "hard to get wrong" is not "checked".
    // =========================================================================================
    {
        using U = SimdVec<PI32,4>;
        alignas( 64 ) PI32 u[ 4 ] = { 1, 2, 3, 0xFFFFFFFFu };
        const double lo[ 4 ] = { 1, 2, 3, 9 };
        const double hi[ 4 ] = { 9, 9, 9, 4294967295.0 };
        CHECK_LANES( lo, 4, min( U::load_aligned( u ), U( 9 ) ) );
        CHECK_LANES( hi, 4, max( U::load_aligned( u ), U( 9 ) ) );
        CHECK( to_bits( asimd::gt( U::load_aligned( u ), U( 1 ) ) ) == 0xeu );

        // the same values read as SIGNED must compare the other way -- 0xFFFFFFFF is -1
        using S = SimdVec<SI32,4>;
        alignas( 64 ) SI32 v[ 4 ] = { 1, 2, 3, -1 };
        CHECK( to_bits( asimd::gt( S::load_aligned( v ), S( 1 ) ) ) == 0x6u );

        using U64 = SimdVec<PI64,2>;
        alignas( 64 ) PI64 u64[ 2 ] = { 3, 0xFFFFFFFFFFFFFFFFull };
        CHECK( to_bits( asimd::gt( U64::load_aligned( u64 ), U64( 9 ) ) ) == 0x2u );

        using U8 = SimdVec<PI8,16>;
        alignas( 64 ) PI8 u8[ 16 ] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0xFFu };
        CHECK( to_bits( asimd::gt( U8::load_aligned( u8 ), U8( 200 ) ) ) == 0x8000u );
    }

    // =========================================================================================
    // 6. AN INTEGER MULTIPLY-ACCUMULATE, which x86 does not have at any feature level
    // =========================================================================================
    {
        using I = SimdVec<SI32,4>;
        const double e[ 4 ] = { 7, 7, 7, 7 };
        CHECK_LANES( e, 4, fma( I( 2 ), I( 3 ), I( 1 ) ) );
        // and at a split width, where it is two `MLA`s
        using J = SimdVec<SI32,8>;
        const double e8[ 8 ] = { 7, 7, 7, 7, 7, 7, 7, 7 };
        CHECK_LANES( e8, 8, fma( J( 2 ), J( 3 ), J( 1 ) ) );
        // negative operands, because `MLA` is signed here and the wrap has to match
        using K = SimdVec<SI32,4>;
        const double en[ 4 ] = { -5, -5, -5, -5 };
        CHECK_LANES( en, 4, fma( K( -2 ), K( 3 ), K( 1 ) ) );
    }

    // =========================================================================================
    // 7. load / store at every width, aligned and unaligned
    // =========================================================================================
    {
        alignas( 64 ) float s[ 32 ];
        for ( int i = 0; i < 32; ++i ) s[ i ] = float( i );
        double e[ 32 ];
        for ( int i = 0; i < 32; ++i ) e[ i ] = i;

        CHECK_LANES( e, 4,  SimdVec<FP32, 4>::load_aligned  ( s ) );
        CHECK_LANES( e, 8,  SimdVec<FP32, 8>::load_aligned  ( s ) );
        CHECK_LANES( e, 16, SimdVec<FP32,16>::load_aligned  ( s ) );
        CHECK_LANES( e, 8,  SimdVec<FP32, 8>::load_unaligned( s ) );
        CHECK_LANES( e, 32, SimdVec<FP32,32>::load_unaligned( s ) );
        CHECK_LANES( e, 5,  SimdVec<FP32, 5>::load_unaligned( s ) );
        CHECK_LANES( e, 8,  SimdVec<FP32, 8, V7>::load_aligned( s ) );
        // the `stream` flavours, which on ARM are the ordinary instruction -- there is no
        // non-temporal SIMD load or store to map them to, and a lane loop would be worse than a
        // missing hint.
        CHECK_LANES( e, 4,  SimdVec<FP32, 4>::load_aligned_stream( s ) );
        CHECK_LANES( e, 8,  SimdVec<FP32, 8>::load_aligned_stream( s ) );
        {   alignas( 64 ) float out[ 8 ] = {};
            SimdVec<FP32,8>::load_aligned( s ).store_aligned_stream( out );
            CHECK( out[ 0 ] == 0 && out[ 7 ] == 7 ); }

        alignas( 64 ) double d[ 8 ];
        for ( int i = 0; i < 8; ++i ) d[ i ] = i;
        CHECK_LANES( e, 2, SimdVec<FP64,2>::load_aligned( d ) );
        CHECK_LANES( e, 4, SimdVec<FP64,4>::load_aligned( d ) );
        CHECK_LANES( e, 8, SimdVec<FP64,8>::load_aligned( d ) );
        CHECK_LANES( e, 2, SimdVec<FP64,2,V7>::load_aligned( d ) );

        alignas( 64 ) SI8 c[ 32 ];
        for ( int i = 0; i < 32; ++i ) c[ i ] = SI8( i );
        CHECK_LANES( e, 16, SimdVec<SI8,16>::load_aligned( c ) );
        CHECK_LANES( e, 32, SimdVec<SI8,32>::load_unaligned( c ) );
    }

    // =========================================================================================
    // 8. gather / scatter, which ARM has no instruction for at all
    //
    // Both keep the generic form -- a lane loop, which is what the hardware would have to do.
    // They still have to WORK, and at a split width the generic form recurses through halves.
    // =========================================================================================
    {
        alignas( 64 ) float data[ 16 ]; for ( int i = 0; i < 16; ++i ) data[ i ] = float( 10 * i );
        alignas( 64 ) SI32  idx [  8 ] = { 7, 6, 5, 4, 3, 2, 1, 0 };
        const double e[ 8 ] = { 70, 60, 50, 40, 30, 20, 10, 0 };
        CHECK_LANES( e, 8, SimdVec<FP32,8>::gather( data, SimdVec<SI32,8>::load_aligned( idx ) ) );
        CHECK_LANES( e, 4, SimdVec<FP32,4>::gather( data, SimdVec<SI32,4>::load_aligned( idx ) ) );

        alignas( 64 ) float out[ 8 ] = {};
        SimdVec<FP32,8>::scatter( out, SimdVec<SI32,8>::load_aligned( idx ),
                                  SimdVec<FP32,8>::iota( 0.f ) );
        CHECK( out[ 7 ] == 0.f && out[ 0 ] == 7.f );
    }

    return report( "test_arm_ops" );
}
