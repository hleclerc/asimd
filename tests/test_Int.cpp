#include <asimd/Int.h>
#include "catch_main.h"
#include "P.h"

using namespace asimd;

template<class T,int a,int o> std::string full_repr( Int<T,a,o> v ) { return "Int<" + std::to_string( a ) + "," + std::to_string( o ) + ">(" + std::to_string( v.get() ) + ")"; }
template<int v> std::string full_repr( N<v> ) { return "N<" + std::to_string( v ) + ">()"; }
std::string full_repr( std::size_t v ) { return std::to_string( v ); }

TEST_CASE( "Integer", "[asimd]" ) {
    SECTION( "+" ) {
        CHECK( full_repr( Int<int,3   >( 10 ) + 17                  ) == "Int<1,0>(27)"  );
        CHECK( full_repr( Int<int,3   >( 12 ) + N<0>()              ) == "Int<3,0>(12)"  );
        CHECK( full_repr( Int<int,3   >( 12 ) + N<4>()              ) == "Int<3,1>(16)"  );
        CHECK( full_repr( Int<int,10,3>( 13 ) + Int<int,15,4>( 19 ) ) == "Int<5,2>(32)"  );
        CHECK( full_repr( Int<int,10,3>( 13 ) - Int<int,15,4>( 19 ) ) == "Int<5,4>(-6)"  );
    }

    SECTION( "*" ) {
        CHECK( full_repr( Int<int,3  >( 10 ) * 17                  ) == "Int<3,0>(170)"  );
        CHECK( full_repr( Int<int,3  >( 12 ) * N<0>()              ) == "N<0>()"         );
        CHECK( full_repr( Int<int,3  >( 12 ) * N<2>()              ) == "Int<6,0>(24)"   );
        CHECK( full_repr( Int<int,3,1>( 13 ) * N<2>()              ) == "Int<6,2>(26)"   );
        CHECK( full_repr( Int<int,3  >( 6  ) * Int<int,4  >( 8 )   ) == "Int<12,0>(48)"  );
        CHECK( full_repr( Int<int,3,2>( 5  ) * Int<int,4  >( 8 )   ) == "Int<4,0>(40)"   );
        CHECK( full_repr( Int<int,4,2>( 6  ) * Int<int,4  >( 8 )   ) == "Int<8,0>(48)"   );
    }
}
