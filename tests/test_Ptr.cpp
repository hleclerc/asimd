#include <asimd/Ptr.h>
#include "catch_main.h"
#include "P.h"

using namespace asimd;

template<class T>
std::string ptr_info( T value ) {
    return std::to_string( value.alignment ) + " " + std::to_string( value.offset );
}

TEST_CASE( "Ptr", "[asimd]" ) {
    SECTION( "known values" ) {
        CHECK( ptr_info( Ptr<float,128>() + N< 0>() ) == "128 0"  );
        CHECK( ptr_info( Ptr<float,128>() + N< 1>() ) == "128 32" );
        CHECK( ptr_info( Ptr<float,128>() + N< 2>() ) == "128 64" );
        CHECK( ptr_info( Ptr<float,128>() + N< 3>() ) == "128 96" );
        CHECK( ptr_info( Ptr<float,128>() + N< 4>() ) == "128 0"  );
        CHECK( ptr_info( Ptr<float,128>() + N<-1>() ) == "128 96" );
        CHECK( ptr_info( Ptr<float,128>() - N< 1>() ) == "128 96" );
    }

    SECTION( "unknown values" ) {
        CHECK( ptr_info( Ptr<float,128,2>() + 1 ) == "32 2" );
        CHECK( ptr_info( Ptr<float, 16,2>() + 1 ) == "16 2" );
    }

    SECTION( "values with known alignment and offset" ) {
        CHECK( ptr_info( Ptr<float,128,32>() + Int<int,1,0>( 0 ) ) == "32 0"  );
        CHECK( ptr_info( Ptr<float,128,32>() + Int<int,2,0>( 0 ) ) == "64 32"  );
        CHECK( ptr_info( Ptr<float,128,32>() + Int<int,4,0>( 0 ) ) == "128 32" );
        CHECK( ptr_info( Ptr<float,128,32>() + Int<int,8,0>( 0 ) ) == "128 32" );

        CHECK( ptr_info( Ptr<float,128,32>() + Int<int,4,1>( 1 ) ) == "128 64" );
        CHECK( ptr_info( Ptr<float,128,32>() + Int<int,4,2>( 2 ) ) == "128 96" );
        CHECK( ptr_info( Ptr<float,128,32>() + Int<int,4,3>( 3 ) ) == "128 0" );
        CHECK( ptr_info( Ptr<float,128,32>() + Int<int,8,3>( 3 ) ) == "128 0" );
    }
}
