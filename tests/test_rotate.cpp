// `rotate_lanes` and `ext_lanes`, value by value, in every spelling: compile-time and runtime
// shift, compile-time and runtime prefix width, free function and method -- against a scalar
// reference, over every (type, width) the two backends split differently. The dispatch grids in
// `test_x86_dispatch.cpp` and `test_arm_dispatch.cpp` say WHICH form ran; this file says whether
// they all agree with the definition:
//
//     rotate_lanes( v, k, n )[ i ] = i < n ? v[ ( i + k ) % n ] : v[ i ]
//     ext_lanes<K>( a, b )[ i ]    = i + K < W ? a[ i + K ] : b[ i + K - W ]
#include <asimd/asimd.h>
#include "check.h"
#include <cstdio>

using namespace asimd;

template<class T,int W> struct Ref {
    static void rotate( T *r, const T *v, int k, int n ) {
        k = ( k % n + n ) % n;
        for ( int i = 0; i < W; ++i ) r[ i ] = i < n ? v[ ( i + k ) % n ] : v[ i ];
    }
    static void ext( T *r, const T *a, const T *b, int K ) {
        for ( int i = 0; i < W; ++i ) r[ i ] = i + K < W ? a[ i + K ] : b[ i + K - W ];
    }
};

/// one ( k, n ) pair, all spellings.
template<int k,int n,class T,int W>
static void one( const char *label, const SimdVec<T,W> &v, const T *src ) {
    T e[ W ];
    Ref<T,W>::rotate( e, src, k, n );
    CHECK_AT( label, lanes_are( rotate_lanes( v, N<k>(), N<n>() ), e, W ) );   // both constant
    CHECK_AT( label, lanes_are( rotate_lanes( v, k, N<n>() ), e, W ) );        // runtime shift
    CHECK_AT( label, lanes_are( rotate_lanes( v, N<k>(), n ), e, W ) );        // runtime width
    CHECK_AT( label, lanes_are( rotate_lanes( v, k, n ), e, W ) );             // both runtime
    CHECK_AT( label, lanes_are( v.rotate_lanes( N<k>(), N<n>() ), e, W ) );
    CHECK_AT( label, lanes_are( v.rotate_lanes( k, n ), e, W ) );
    if constexpr ( n == W ) {
        CHECK_AT( label, lanes_are( rotate_lanes( v, N<k>() ), e, W ) );
        CHECK_AT( label, lanes_are( rotate_lanes( v, k ), e, W ) );
        CHECK_AT( label, lanes_are( v.rotate_lanes( N<k>() ), e, W ) );
        CHECK_AT( label, lanes_are( v.rotate_lanes( k ), e, W ) );
    }
}

/// every k in [-n, 2n), for one n
template<int n,class T,int W,int... ks>
static void shifts( const char *label, const SimdVec<T,W> &v, const T *src, std::integer_sequence<int,ks...> ) {
    ( one<int( ks ) - n,n,T,W>( label, v, src ), ... );
}

template<class T,int W,int... ns>
static void widths( const char *label, const SimdVec<T,W> &v, const T *src, std::integer_sequence<int,ns...> ) {
    ( shifts<int( ns ) + 1,T,W>( label, v, src, std::make_integer_sequence<int,3 * ( int( ns ) + 1 )>() ), ... );
}

template<class T,int W,int... Ks>
static void exts( const char *label, const SimdVec<T,W> &a, const SimdVec<T,W> &b, const T *sa, const T *sb,
                  std::integer_sequence<int,Ks...> ) {
    ( [ & ] {
        T e[ W ];
        Ref<T,W>::ext( e, sa, sb, Ks );
        CHECK_AT( label, lanes_are( ext_lanes( a, b, N<Ks>() ), e, W ) );
        CHECK_AT( label, lanes_are( a.ext_lanes( b, N<Ks>() ), e, W ) );
    }(), ... );
}

template<class T,int W>
static void grid( const char *label ) {
    alignas( 64 ) T sa[ W ], sb[ W ];
    for ( int i = 0; i < W; ++i ) { sa[ i ] = T( 10 + i ); sb[ i ] = T( 50 + i ); }
    const auto a = SimdVec<T,W>::load_aligned( sa ), b = SimdVec<T,W>::load_aligned( sb );
    widths<T,W>( label, a, sa, std::make_integer_sequence<int,W>() );
    exts<T,W>( label, a, b, sa, sb, std::make_integer_sequence<int,W + 1>() );
}

int main() {
    // the widths that matter: one register, a split of two, a split of four, and the ones that
    // are not a power of two (a 4 + 1 split, where the halves differ).
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
    grid<PI32  ,8 >( "PI32 x 8" );
    grid<SI64  ,2 >( "SI64 x 2" );
    grid<SI64  ,4 >( "SI64 x 4" );
    grid<SI64  ,8 >( "SI64 x 8" );
    grid<SI16  ,8 >( "SI16 x 8" );
    grid<SI16  ,16>( "SI16 x 16" );
    grid<SI8   ,16>( "SI8 x 16" );
    grid<SI8   ,32>( "SI8 x 32" );
    grid<PI16  ,32>( "PI16 x 32" );

    // the identity spellings
    {
        alignas( 64 ) float s[ 8 ] = { 1, 2, 3, 4, 5, 6, 7, 8 };
        const auto v = SimdVec<float,8>::load_aligned( s );
        CHECK_LANES( s, 8, rotate_lanes( v, N<0>() ) );
        CHECK_LANES( s, 8, rotate_lanes( v, N<8>() ) );
        CHECK_LANES( s, 8, rotate_lanes( v, 16 ) );
        CHECK_LANES( s, 8, rotate_lanes( v, N<3>(), N<1>() ) );
        CHECK_LANES( s, 8, ext_lanes( v, v, N<0>() ) );
        CHECK_LANES( s, 8, ext_lanes( v, v, N<8>() ) );
    }
    return report( "test_rotate" );
}
