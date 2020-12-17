#include <asimd/architectures/X86Cpu.h>
#include <asimd/SimdRange.h>
#include "catch_main.h"
#include "P.h"

using namespace asimd;

template<class T>
std::string int_info( T value ) {
    return std::to_string( value.get() ) + " al:" + std::to_string( value.alignment ) + " " + std::to_string( value.offset );
}

template<int size,class Beg>
void check_simd_range( N<size>, Beg beg, int end, std::vector<int> expected_indices ) {
    std::vector<int> indices, simd_sizes, alignments;
    SimdRange<size>::for_each( beg, end, [&]( auto index, auto simd_size ) {
        P( int_info( index ) );
        indices.push_back( index.get() );
        simd_sizes.push_back( simd_size.get() );
        CHECK( index.offset == 0 );
    } );
    for( std::size_t i = 1; i < simd_sizes.size(); ++i )
        CHECK( simd_sizes[ i - 1 ] == indices[ i ] - indices[ i - 1 ] );
    CHECK( indices == expected_indices );
}

TEST_CASE( "SimdRange", "[asimd]" ) {
    check_simd_range( N<4>(), Int<int,4,0>( 0 ), 11, { 0, 4, 8, 10 } );
    check_simd_range( N<4>(), Int<int,4,2>( 2 ), 11, { 2, 4, 8, 10 } );
}
