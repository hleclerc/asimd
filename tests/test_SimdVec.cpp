#define ASIMD_DEBUG_ON_OP( NAME, COND ) \
    ops += std::string( ops.empty() ? "" : " " ) + NAME + "[" + COND + "]";
#include <string>
std::string ops;

#include <asimd/architectures/X86Cpu.h>
#include <asimd/NbSimdRegisters.h>
#include <asimd/SimdVec.h>
#include "catch_main.h"
#include "P.h"

using namespace asimd;

template<class V>
bool equal( const V &a, const V &b ) {
    if ( a.size() != b.size() )
        return false;
    for( std::size_t i = 0; i < a.size(); ++i )
        if ( a[ i ] != b[ i ] )
            return false;
    return true;
}

TEST_CASE( "SimdVec", "[asimd]" ) {
    using Arch = X86Cpu<8,features::SSE2>;
    using V4 = SimdVec<int,4,Arch>;
    using V5 = SimdVec<int,5,Arch>;
    using V8 = SimdVec<int,8,Arch>;
    using V9 = SimdVec<int,9,Arch>;

    CHECK( SimdSize<int,Arch>::value == 4 );

    SECTION( "load unaligned" ) {
        ops.clear();
        int d[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
        V9 v = V9::load_unaligned( d );

        for( int i = 0; i < v.size(); ++i )
            CHECK( i == v[ i ] );
        CHECK( ops == "load_unaligned[SSE2] load_unaligned[SSE2]" );
    }

    SECTION( "load with Ptr" ) {
        ops.clear();

        alignas( 128 ) int d[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        Ptr<int,128,0> p( d );

        V8 w = V8::load( p + N<1>() );
        V8 v = V8::load( p + N<0>() );
        V4 u = V4::load( p + Int<int,4>( 4 ) );
        V4 s = V4::load( p + Int<int,4,1>( 5 ) );

        CHECK( equal( w, { 2, 3, 4, 5, 6, 7, 8, 9 } ) );
        CHECK( equal( v, { 1, 2, 3, 4, 5, 6, 7, 8 } ) );
        CHECK( equal( u, { 5, 6, 7, 8 } ) );
        CHECK( equal( s, { 6, 7, 8, 9 } ) );
        CHECK( ops == "load_unaligned[SSE2] load_unaligned[SSE2] load_aligned[SSE2] load_aligned[SSE2] load_aligned[SSE2] load_unaligned[SSE2]" );
    }

    SECTION( "arithmetic operations" ) {
        V5 v( 10 ), w = V5::iota();

        CHECK( equal( v + w, { 10, 11, 12, 13, 14 } ) );
        CHECK( equal( v - w, { 10,  9,  8,  7,  6 } ) );
        CHECK( equal( v * w, {  0, 10, 20, 30, 40 } ) );
    }

    SECTION( "load with *" ) {
        ops.clear();

        alignas( 128 ) int d[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        V8 w = V8::load_unaligned( d + 1 );
        V8 v = V8::load_aligned( d );

        CHECK( equal( w, { 2, 3, 4, 5, 6, 7, 8, 9 } ) );
        CHECK( equal( v, { 1, 2, 3, 4, 5, 6, 7, 8 } ) );
        CHECK( ops == "load_unaligned[SSE2] load_unaligned[SSE2] load_aligned[SSE2] load_aligned[SSE2]" );
    }

    SECTION( "NbSimdRegisters" ) {
        CHECK( NbSimdRegisters<int,4,X86Cpu<64,features::SSE2  >>::value ==  8 );
        CHECK( NbSimdRegisters<int,4,X86Cpu<64,features::AVX   >>::value == 16 );
        CHECK( NbSimdRegisters<int,4,X86Cpu<64,features::AVX2  >>::value == 16 );
        CHECK( NbSimdRegisters<int,4,X86Cpu<64,features::AVX512>>::value == 32 );

        CHECK( NbSimdRegisters<int,8,X86Cpu<64,features::SSE2  >>::value ==  4 );
    }
}
