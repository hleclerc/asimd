// `load_partial` / `store_partial`: the lanes of a set and NOT ONE BYTE OUTSIDE THEM.
//
// The values are the easy half. The contract is about the bytes that are not touched, and no
// value can show that -- so the buffer is placed against a PROT_NONE page, twice: once with the
// last lane of the set as the last lane before the guard (a read or write of any lane past it
// is a fault), once with the first lane of the set as the first lane after a guard (any lane
// before it faults). Lanes inside the buffer but outside the set are poisoned before a store
// and checked afterwards.
#include <asimd/asimd.h>
#include "check.h"
#include <sys/mman.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>

using namespace asimd;

struct GuardedPage {
    GuardedPage() {
        page = size_t( sysconf( _SC_PAGESIZE ) );
        base = (char *) mmap( nullptr, 3 * page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0 );
        if ( base == MAP_FAILED ) { perror( "mmap" ); exit( 1 ); }
        mprotect( base, page, PROT_NONE );
        mprotect( base + 2 * page, page, PROT_NONE );
    }
    ~GuardedPage() { munmap( base, 3 * page ); }
    char  *rw_beg() const { return base + page; }
    char  *rw_end() const { return base + 2 * page; }
    size_t page;
    char  *base;
};

static GuardedPage &guarded() { static GuardedPage g; return g; }

template<class T,int W,class S>
static void at( const char *label, const S &s, T *ptr, int first, int last ) {
    using V = SimdVec<T,W>;
    // ---- load: the lanes of the set come back; nothing else is read (or this faults)
    for ( int i = first; i <= last; ++i ) ptr[ i ] = T( 10 + i );
    const V v = V::load_partial( ptr, s );
    alignas( 64 ) T got[ W ];
    V::store_unaligned( got, v );
    bool ok = true;
    for ( int i = 0; i < W; ++i )
        if ( s.has( i ) && got[ i ] != T( 10 + i ) ) {
            printf( "  info   lane %d of %d is %g, expected %g -- see the FAIL below\n", i, W, double( got[ i ] ), double( 10 + i ) );
            ok = false;
        }
    CHECK_AT( label, ok );

    // ---- store: the lanes of the set are written; the others -- inside the buffer -- are not
    alignas( 64 ) T src[ W ];
    for ( int i = 0; i < W; ++i ) src[ i ] = T( 50 + i );
    for ( int i = first; i <= last; ++i ) ptr[ i ] = T( 99 );
    V::store_partial( ptr, V::load_unaligned( src ), s );
    ok = true;
    for ( int i = first; i <= last; ++i ) {
        const T want = s.has( i ) ? T( 50 + i ) : T( 99 );
        if ( ptr[ i ] != want ) {
            printf( "  info   memory lane %d is %g, expected %g -- see the FAIL below\n", i, double( ptr[ i ] ), double( want ) );
            ok = false;
        }
    }
    CHECK_AT( label, ok );
    // the member spelling
    for ( int i = first; i <= last; ++i ) ptr[ i ] = T( 99 );
    V::load_unaligned( src ).store_partial( ptr, s );
    ok = true;
    for ( int i = first; i <= last; ++i ) if ( ptr[ i ] != ( s.has( i ) ? T( 50 + i ) : T( 99 ) ) ) ok = false;
    CHECK_AT( label, ok );
}

template<class T,int W,class S>
static void one( const char *label, const S &s ) {
    int first = -1, last = -1;
    for ( int i = 0; i < W; ++i ) if ( s.has( i ) ) { if ( first < 0 ) first = i; last = i; }
    if ( first < 0 ) return;
    GuardedPage &g = guarded();
    // the last lane of the set is the last lane before the guard
    at<T,W>( label, s, (T *) g.rw_end() - ( last + 1 ), first, last );
    // the first lane of the set is the first lane after the guard
    at<T,W>( label, s, (T *) g.rw_beg() - first, first, last );
}

template<class T,int W,int B,int E>
static void shapes( const char *label ) {
    char l[ 64 ]; snprintf( l, sizeof l, "%s [%d,%d)", label, B, E );
    one<T,W>( l, LaneRange<B,E>() );
    one<T,W>( l, LaneRange<B>( E ) );
    one<T,W>( l, LaneRange( B, E ) );
    one<T,W>( l, LaneMask<>( LaneRange<B,E>().bits( 64 ) ) );
    one<T,W>( l, LaneMask<B,E>( LaneRange<B,E>().bits( 64 ) ) );
    PI64 holes = 0; for ( int i = B; i < E; i += 2 ) holes |= PI64( 1 ) << i;
    one<T,W>( l, LaneMask<B,E>( holes ) );
    one<T,W>( l, LaneMask<>( holes ) );
    if constexpr ( B == 0 ) {
        using V = SimdVec<T,W>;
        // the `int n` spelling, against the guard
        GuardedPage &g = guarded();
        T *ptr = (T *) g.rw_end() - E;
        for ( int i = 0; i < E; ++i ) ptr[ i ] = T( 10 + i );
        const V v = V::load_partial( ptr, E );
        alignas( 64 ) T got[ W ]; V::store_unaligned( got, v );
        bool ok = true; for ( int i = 0; i < E; ++i ) ok &= got[ i ] == T( 10 + i );
        CHECK_AT( l, ok );
        v.store_partial( ptr, E );
        ok = true; for ( int i = 0; i < E; ++i ) ok &= ptr[ i ] == T( 10 + i );
        CHECK_AT( l, ok );
    }
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
    if constexpr ( W >= 8 ) { shapes<T,W,0,3>( label ); shapes<T,W,5,W>( label ); shapes<T,W,3,5>( label ); shapes<T,W,0,W-1>( label ); }
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
    grid<PI32  ,8 >( "PI32 x 8" );
    grid<SI64  ,2 >( "SI64 x 2" );
    grid<SI64  ,4 >( "SI64 x 4" );
    grid<SI64  ,8 >( "SI64 x 8" );
    grid<SI16  ,8 >( "SI16 x 8" );
    grid<SI16  ,16>( "SI16 x 16" );
    grid<SI16  ,32>( "SI16 x 32" );
    grid<SI8   ,16>( "SI8 x 16" );
    grid<SI8   ,64>( "SI8 x 64" );
    return report( "test_partial" );
}
