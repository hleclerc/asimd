// #include <cstdint>
// #include "HaD.h"

namespace asimd {

template<class I> inline constexpr
I prev_pow_2( I v ) {
    return 1 << ( sizeof( I ) * 8 - __builtin_clz( v - 1 ) - 1 );
}

} // namespace asimd
