#pragma once

// A small harness. The upstream tests use Catch2; pulling in that dependency to assert that a
// vector addition returns 3 would be out of proportion.
//
// Three verdicts, not two. `CHECK` is an assertion. `KNOWN_BROKEN` records a bug that is
// DIAGNOSED AND NOT YET FIXED: it prints, it does not fail the build, and it turns into a loud
// XPASS the day the bug is fixed -- which is the signal to promote it to a plain `CHECK`. A test
// suite that goes red on day one gets ignored; one that lists its own to-do list does not.

#include <cmath>
#include <cstdio>

inline int nb_checks = 0, nb_failures = 0, nb_known_broken = 0, nb_fixed = 0;

inline void check( bool ok, const char *what, const char *file, int line ) {
    ++nb_checks;
    if ( ! ok ) { ++nb_failures; printf( "  FAIL   %s:%d  %s\n", file, line, what ); }
}

// Variadic on purpose: a template argument list carries commas, and the preprocessor does not
// respect angle brackets. `CHECK( SimdVec<float,8>::size() == 8 )` would otherwise not compile.
#define CHECK( ... ) check( ( __VA_ARGS__ ), #__VA_ARGS__, __FILE__, __LINE__ )

/// same, with a label saying WHICH instantiation failed -- a grid test runs the same line for
/// every (type, width), so the line number alone does not identify the failure.
inline void check_at( bool ok, const char *what, const char *label, const char *file, int line ) {
    ++nb_checks;
    if ( ! ok ) { ++nb_failures; printf( "  FAIL   %s:%d  [%s]  %s\n", file, line, label, what ); }
}

#define CHECK_AT( label, ... ) check_at( ( __VA_ARGS__ ), #__VA_ARGS__, label, __FILE__, __LINE__ )

/// An assertion that is EXPECTED TO FAIL, with the reason. See the note above.
inline void check_known_broken( bool ok, const char *what, const char *file, int line,
                                const char *why ) {
    if ( ok ) {
        ++nb_fixed;
        printf( "  XPASS  %s:%d  %s\n         -> \"%s\" now holds: make this a plain CHECK.\n",
                file, line, what, why );
    } else {
        ++nb_known_broken;
        printf( "  broken %s:%d  %s\n         -> %s\n", file, line, what, why );
    }
}

#define KNOWN_BROKEN( why, ... ) check_known_broken( ( __VA_ARGS__ ), #__VA_ARGS__, __FILE__, __LINE__, why )

/// compares a vector against a list of expected values.
///
/// IT REPORTS THE LANE, which it did not, and that is most of what this helper is for. It backs
/// hundreds of grid assertions through `CHECK_AT( label, lanes_are( ... ) )`, and a failure used
/// to print the label and the expression and nothing else -- "something among these lanes
/// differs". Twice in one week that was the whole of the information available about a failure on
/// a compiler that could not be run locally, and twice it cost a full CI round trip to get the
/// number that was one `printf` away.
///
/// `check_lanes`, just below, has always reported it; `lanes_are` is the variant used where the
/// caller also has a label, and it was the one that said nothing. The `info` line is printed
/// BEFORE the `FAIL` line it belongs to, because the comparison necessarily runs first -- and the
/// prefix matches the one the test harnesses' filters pass.
template<class V, class T>
inline bool lanes_are( const V &v, const T *expected, int n ) {
    alignas( 64 ) typename V::T got[ 64 ] = {};
    V::store_unaligned( got, v );
    for ( int i = 0; i < n; ++i )
        if ( ! ( std::abs( double( got[ i ] ) - double( expected[ i ] ) ) < 1e-6 ) ) {
            printf( "  info   lane %d of %d is %g, expected %g -- see the FAIL below\n",
                    i, n, double( got[ i ] ), double( expected[ i ] ) );
            return false;
        }
    return true;
}

template<class V, class T>
inline void check_lanes( const V &v, const T *expected, int n, const char *what,
                         const char *file, int line ) {
    alignas( 64 ) typename V::T got[ 64 ] = {};
    v.store_unaligned( got );
    for ( int i = 0; i < n; ++i )
        if ( ! ( std::abs( double( got[ i ] ) - double( expected[ i ] ) ) < 1e-6 ) ) {
            ++nb_checks; ++nb_failures;
            printf( "  FAIL   %s:%d  %s : lane %d is %g, expected %g\n",
                    file, line, what, i, double( got[ i ] ), double( expected[ i ] ) );
            return;
        }
    ++nb_checks;
}

/// `CHECK_LANES( expected, n, <the vector expression> )` -- the expression comes LAST so that the
/// commas of its template arguments fall inside `__VA_ARGS__` rather than splitting the macro.
#define CHECK_LANES( expected, n, ... ) \
    check_lanes( ( __VA_ARGS__ ), expected, n, #__VA_ARGS__, __FILE__, __LINE__ )

/// idem, for a vector whose lanes are only known to be right up to `n`, expected as doubles.
#define CHECK_LANES_KNOWN_BROKEN( expected, n, why, ... ) \
    check_known_broken( lanes_are( ( __VA_ARGS__ ), expected, n ), #__VA_ARGS__, __FILE__, __LINE__, why )

inline int report( const char *name ) {
    printf( "%-20s %d checks, %d failures", name, nb_checks, nb_failures );
    if ( nb_known_broken ) printf( ", %d known broken", nb_known_broken );
    if ( nb_fixed )        printf( ", %d NEWLY FIXED", nb_fixed );
    printf( "\n" );
    return nb_failures ? 1 : 0;
}
