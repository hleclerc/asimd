// LANE SETS, value by value and register by register.
//
// Two questions. Are the lanes INSIDE the set right -- for every operation that takes one, in
// every shape of set (static range, half-static range, dynamic range, mask)? And are the
// registers OUTSIDE the set actually skipped -- which no value can show, so the register forms
// are COUNTED through `ASIMD_DEBUG_ON_OP`: `add( a, b, LaneRange<0,3>() )` on a vector of two
// registers must issue one `add`, not two.
#define ASIMD_DEBUG_ON_OP( NAME, COND, FUNC ) ++nb_register_ops;
#include <cstdio>
inline int nb_register_ops = 0;

#include <asimd/asimd.h>
#include "check.h"
#include <cmath>

using namespace asimd;

template<class T,int W,class S>
static void one( const char *label, const S &s ) {
    using V = SimdVec<T,W>;
    alignas( 64 ) T sa[ W ], sb[ W ], sc[ W ];
    for ( int i = 0; i < W; ++i ) { sa[ i ] = T( 3 + i ); sb[ i ] = T( 1 + ( i * 7 ) % 5 ); sc[ i ] = T( 2 * i ); }
    const V a = V::load_aligned( sa ), b = V::load_aligned( sb ), c = V::load_aligned( sc );

    // the lanes in the set, checked one by one; the others are not looked at
    auto lanes_in_set = [ & ]( const V &v, auto ref ) {
        alignas( 64 ) T got[ W ];
        V::store_aligned( got, v );
        for ( int i = 0; i < W; ++i )
            if ( s.has( i ) && ! ( std::abs( double( got[ i ] ) - double( ref( i ) ) ) < 1e-6 ) ) {
                printf( "  info   lane %d of %d is %g, expected %g -- see the FAIL below\n", i, W, double( got[ i ] ), double( ref( i ) ) );
                return false;
            }
        return true;
    };
    CHECK_AT( label, lanes_in_set( add( a, b, s ), [ & ]( int i ) { return sa[ i ] + sb[ i ]; } ) );
    CHECK_AT( label, lanes_in_set( sub( a, b, s ), [ & ]( int i ) { return sa[ i ] - sb[ i ]; } ) );
    CHECK_AT( label, lanes_in_set( mul( a, b, s ), [ & ]( int i ) { return sa[ i ] * sb[ i ]; } ) );
    CHECK_AT( label, lanes_in_set( div( a, b, s ), [ & ]( int i ) { return sa[ i ] / sb[ i ]; } ) );
    CHECK_AT( label, lanes_in_set( min( a, b, s ), [ & ]( int i ) { return sa[ i ] < sb[ i ] ? sa[ i ] : sb[ i ]; } ) );
    CHECK_AT( label, lanes_in_set( max( a, b, s ), [ & ]( int i ) { return sa[ i ] > sb[ i ] ? sa[ i ] : sb[ i ]; } ) );
    CHECK_AT( label, lanes_in_set( fma( a, b, c, s ), [ & ]( int i ) { return sa[ i ] * sb[ i ] + sc[ i ]; } ) );
    CHECK_AT( label, lanes_in_set( a.add( b, s ), [ & ]( int i ) { return sa[ i ] + sb[ i ]; } ) );
    CHECK_AT( label, lanes_in_set( a.fma( b, c, s ), [ & ]( int i ) { return sa[ i ] * sb[ i ] + sc[ i ]; } ) );

    // comparisons and selections. `sc > sa` flips at i = 3
    CHECK_AT( label, lanes_in_set( select( gt( c, a, s ), a, b, s ), [ & ]( int i ) { return sc[ i ] > sa[ i ] ? sa[ i ] : sb[ i ]; } ) );
    CHECK_AT( label, lanes_in_set( select( c > a, a, b, s ), [ & ]( int i ) { return sc[ i ] > sa[ i ] ? sa[ i ] : sb[ i ]; } ) );
    CHECK_AT( label, lanes_in_set( select( ge( c, a, s ), a, b, s ), [ & ]( int i ) { return sc[ i ] >= sa[ i ] ? sa[ i ] : sb[ i ]; } ) );

    // reductions: the set is part of the answer
    T ref_sum = 0; PI64 ref_bits = 0; bool ref_any = false, ref_all = true;
    for ( int i = 0; i < W; ++i ) if ( s.has( i ) ) {
        ref_sum += sa[ i ];
        const bool gtr = sc[ i ] > sa[ i ];
        if ( gtr ) ref_bits |= PI64( 1 ) << i;
        ref_any |= gtr; ref_all &= gtr;
    }
    CHECK_AT( label, std::abs( double( sum( a, s ) ) - double( ref_sum ) ) < 1e-6 );
    CHECK_AT( label, std::abs( double( a.sum( s ) ) - double( ref_sum ) ) < 1e-6 );
    CHECK_AT( label, to_bits( gt( c, a ), s ) == ref_bits );
    CHECK_AT( label, to_bits( c > a, s ) == ref_bits );
    CHECK_AT( label, any( gt( c, a ), s ) == ref_any );
    CHECK_AT( label, all( gt( c, a ), s ) == ref_all );
    CHECK_AT( label, any( c > a, s ) == ref_any );
    CHECK_AT( label, all( c > a, s ) == ref_all );
}

/// every shape of set over the same lanes
template<class T,int W,int B,int E>
static void shapes( const char *label ) {
    char l[ 64 ]; snprintf( l, sizeof l, "%s [%d,%d)", label, B, E );
    one<T,W>( l, LaneRange<B,E>() );
    one<T,W>( l, LaneRange<B>( E ) );
    one<T,W>( l, LaneRange( B, E ) );
    one<T,W>( l, LaneMask<>( LaneRange<B,E>().bits( 64 ) ) );
    one<T,W>( l, LaneMask<B,E>( LaneRange<B,E>().bits( 64 ) ) );
    // a mask with holes, inside the same hull
    PI64 holes = 0; for ( int i = B; i < E; i += 2 ) holes |= PI64( 1 ) << i;
    one<T,W>( l, LaneMask<B,E>( holes ) );
    one<T,W>( l, LaneMask<>( holes ) );
}

template<class T,int W>
static void grid( const char *label ) {
    shapes<T,W,0,W>( label );
    shapes<T,W,0,1>( label );
    shapes<T,W,0,W/2>( label );
    shapes<T,W,0,W/2+1>( label );
    shapes<T,W,W/2,W>( label );
    shapes<T,W,1,W-1>( label );
    shapes<T,W,W-1,W>( label );
    if constexpr ( W >= 8 ) { shapes<T,W,0,3>( label ); shapes<T,W,5,W>( label ); shapes<T,W,3,5>( label ); }
}

int main() {
    grid<float ,2 >( "float x 2" );
    grid<float ,4 >( "float x 4" );
    grid<float ,8 >( "float x 8" );
    grid<float ,16>( "float x 16" );
    grid<float ,5 >( "float x 5" );
    grid<float ,6 >( "float x 6" );
    grid<double,2 >( "double x 2" );
    grid<double,4 >( "double x 4" );
    grid<double,8 >( "double x 8" );
    grid<double,3 >( "double x 3" );
    grid<SI32  ,4 >( "SI32 x 4" );
    grid<SI32  ,8 >( "SI32 x 8" );
    grid<SI32  ,16>( "SI32 x 16" );
    grid<SI64  ,4 >( "SI64 x 4" );
    grid<SI16  ,16>( "SI16 x 16" );

    // ---- ARE THE REGISTERS SKIPPED? Counted, on the widths that are several registers here.
    {
        using V = SimdVec<float,4>;
        const V a( 1.f ), b( 2.f );
        nb_register_ops = 0; (void) add( a, b );                                const int whole = nb_register_ops;
        nb_register_ops = 0; (void) add( a, b, LaneRange<0,3>() );              const int part  = nb_register_ops;
        printf( "  float x 4:  add = %d register op(s), add on [0,3) = %d\n", whole, part );
        CHECK( part == whole );   // one register: nothing to skip, nothing added either
    }
    {
        using V = SimdVec<float,16>;
        const V a( 1.f ), b( 2.f );
        nb_register_ops = 0; (void) add( a, b );                                const int whole = nb_register_ops;
        nb_register_ops = 0; (void) add( a, b, LaneRange<0,3>() );              const int first = nb_register_ops;
        nb_register_ops = 0; (void) add( a, b, LaneRange<13,16>() );            const int last  = nb_register_ops;
        nb_register_ops = 0; (void) add( a, b, LaneRange<0,9>() );              const int most  = nb_register_ops;
        nb_register_ops = 0; (void) add( a, b, LaneRange<0>( 3 ) );             const int dyn3  = nb_register_ops;
        nb_register_ops = 0; (void) add( a, b, LaneMask<0,4>( 0x5 ) );          const int msk   = nb_register_ops;
        nb_register_ops = 0; (void) add( a, b, LaneRange( 0, 16 ) );            const int dynw  = nb_register_ops;
        printf( "  float x 16: add = %d register op(s); on [0,3) = %d, [13,16) = %d, [0,9) = %d, [0,n=3) = %d, mask in [0,4) = %d, [0,n=16) = %d\n",
                whole, first, last, most, dyn3, msk, dynw );
        if ( whole >= 4 ) {                       // four registers on NEON / SSE, two on AVX2, one on AVX-512
            CHECK( first == whole / 4 );          // static prefix: one register, recursively
            CHECK( last  == whole / 4 );          // static suffix: same, the other end
            CHECK( msk   == whole / 4 );          // the mask's hull is what prunes
            CHECK( dynw  == whole );
        }
        if ( whole == 4 ) {
            CHECK( most == 3 );                   // [0,9): v0 whole (2) + v1's first half (1)
            CHECK( dyn3 == 2 );                   // dynamic end: one branch at the top, where the halves are
                                                  // splits, skips two registers; none below, where they are single
        }
        if ( whole == 2 ) {
            CHECK( first == 1 && last == 1 && msk == 1 );
            CHECK( dyn3 == 2 );                   // dynamic: the halves are single registers, no branch
        }
    }
    {
        // the reductions prune too, and mask at the finest register
        using V = SimdVec<float,8>;
        alignas( 64 ) float s[ 8 ] = { 1, 2, 3, 4, 5, 6, 7, 8 };
        const V v = V::load_aligned( s );
        CHECK( sum( v, LaneRange<0,3>() ) == 6 );
        CHECK( sum( v, LaneRange<0,4>() ) == 10 );
        CHECK( sum( v, LaneRange<2,6>() ) == 18 );
        CHECK( sum( v, LaneRange<0>( 5 ) ) == 15 );
        CHECK( sum( v, LaneMask<>( 0b10000001 ) ) == 9 );
        CHECK( sum( v, LaneRange<4,8>() ) == 26 );
        CHECK( sum( v ) == 36 );
    }
    return report( "test_lanes" );
}
