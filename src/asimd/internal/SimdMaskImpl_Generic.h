#pragma once

#include <cstdint>
#include <bitset>
#include "HaD.h"
#include "S.h"

namespace asimd {
namespace SimdMaskImpl {

template<class I> inline constexpr
I prew_pow_2( I v ) {
    return 1 << ( sizeof( I ) * 8 - __builtin_clz( v - 1 ) - 1 );
}

// Impl ---------------------------------------------------------
template<int size,class Arch,class Enable=void>
struct Impl {
    static constexpr int split_size_0 = prew_pow_2( size );
    static constexpr int split_size_1 = size - split_size_0;
    struct Split {
        Impl<split_size_0,Arch> v0;
        Impl<split_size_1,Arch> v1;
    };
    union {
        std::bitset<size> values;
        Split split;
    } data;
};

template<class Arch,class Enable>
struct Impl<8,Arch,Enable> {
    union {
        std::bitset<8> values;
    } data;
};

template<class Arch,class Enable>
struct Impl<1,Arch,Enable> {
    union {
        std::bitset<1> values;
    } data;
};

/// Helper to make Impl with a register
#define SIMD_VEC_IMPL_REG( COND, T, SIZE, TREG ) \
    template<class Arch> \
    struct Impl<T,SIZE,Arch,typename std::enable_if<Arch::template Has<features::COND>::value>::type> { \
        static constexpr int split_size_0 = prew_pow_2( SIZE ); \
        static constexpr int split_size_1 = SIZE - split_size_0; \
        struct Split { \
            Impl<T,split_size_0,Arch> v0; \
            Impl<T,split_size_1,Arch> v1; \
        }; \
        union { \
            T     values[ SIZE ]; \
            Split split; \
            TREG  reg; \
        } data; \
    }

// at ------------------------------------------------------------------------
template<int size,class Arch> HaD
bool at( const Impl<size,Arch> &vec, int i ) {
    return vec.data.values[ i ];
}


} // namespace SimdMaskImpl
} // namespace asimd

