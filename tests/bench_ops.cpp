// TIMINGS FOR THE CROSS-LANE OPERATIONS, where counting instructions is a poor proxy.
//
// `no_vecext.sh` counts instructions, which answers "did the dispatch table fire". It cannot
// answer "was that worth it": the generic `permute` stores the vector to the stack, indexes it
// there and reloads it, and the reload is a store-to-load forwarding stall that no instruction
// count shows. At SSE4.1 the split form and the generic form come out at the SAME instruction
// count and a factor of five apart in time.
//
// Two traps this file exists to avoid, both met while writing it:
//
//   FOLDABLE INDICES. With indices the compiler can see, `__builtin_shuffle` becomes a CONSTANT
//   shuffle with an immediate, and the comparison stops being between two implementations of the
//   same operation. Hence `volatile int seed`.
//
//   THE FREQUENCY RAMP. Whichever variant runs first pays it. On this box that alone looked like
//   a 1.9x regression on a loop the disassembler said was identical instruction for instruction.
//   Hence the warm-up and the best-of-five.
#include <asimd/asimd.h>

#include <chrono>
#include <cstdio>

using namespace asimd;

volatile int seed = 5;

/// a sink the optimiser cannot see through, without inline asm -- which is gcc/clang only, and
/// this file has to build under MSVC like everything else here.
template<class T> void keep( T v ) { static volatile T sink; sink = v; (void) sink; }

template<class F>
static double best_of( F &&f ) {
    double best = 1e30;
    for ( int rep = 0; rep < 6; ++rep ) {
        auto t0 = std::chrono::steady_clock::now();
        auto acc = f();
        auto t1 = std::chrono::steady_clock::now();
        keep( acc );
        const double ns = std::chrono::duration<double,std::nano>( t1 - t0 ).count();
        if ( rep && ns < best ) best = ns;                      // rep 0 is the warm-up
    }
    return best;
}

int main() {
    constexpr int N   = SimdSize<float,NativeCpu>::value;
    constexpr int REP = 3000000;
    using V = SimdVec<float,N>;
    using I = SimdVec<SI32,N>;
    using W = SimdVec<float,2*N>;                               // a width that must split
    using J = SimdVec<SI32,2*N>;

    alignas( 64 ) float s[ 2 * N ];
    alignas( 64 ) SI32  x[ 2 * N ];
    for ( int i = 0; i < 2 * N; ++i ) { s[ i ] = float( i ); x[ i ] = SI32( ( i * seed + 3 ) % ( 2 * N ) ); }

    printf( "  %s, %d lanes native\n", NativeCpu::name().c_str(), N );
    printf( "  %-22s %10s\n", "", "ns/op" );

    const V v = V::load_aligned( s ); const I id = I::load_aligned( x );
    const W w = W::load_aligned( s ); const J jd = J::load_aligned( x );

    printf( "  %-22s %10.2f\n", "permute, native width",
        best_of( [ & ] { V a = v; float r = 0; for ( int k = 0; k < REP; ++k ) { a = permute( a, id ); r += a[ 0 ]; } return r; } ) / REP );
    printf( "  %-22s %10.2f\n", "permute, 2x native",
        best_of( [ & ] { W a = w; float r = 0; for ( int k = 0; k < REP; ++k ) { a = permute( a, jd ); r += a[ 0 ]; } return r; } ) / REP );
    printf( "  %-22s %10.2f\n", "to_bits, native width",
        best_of( [ & ] { PI64 r = 0; for ( int k = 0; k < REP; ++k ) r += to_bits( gt( v, V( float( k & 7 ) ) ) ); return r; } ) / REP );
    printf( "  %-22s %10.2f\n", "to_bits, 2x native",
        best_of( [ & ] { PI64 r = 0; for ( int k = 0; k < REP; ++k ) r += to_bits( gt( w, W( float( k & 7 ) ) ) ); return r; } ) / REP );
    printf( "  %-22s %10.2f\n", "sum, native width",
        best_of( [ & ] { float r = 0; for ( int k = 0; k < REP; ++k ) r += ( v + V( float( k ) ) ).sum(); return r; } ) / REP );
    printf( "  %-22s %10.2f\n", "sum, 2x native",
        best_of( [ & ] { float r = 0; for ( int k = 0; k < REP; ++k ) r += ( w + W( float( k ) ) ).sum(); return r; } ) / REP );
    return 0;
}
