#pragma once

#include "ASIMD_DEBUG_ON_OP.h" // IWYU pragma: export

#include "../support/prev_pow_2.h"
#include "../support/BitVec.h"
#include "../support/HaD.h"

namespace asimd {
namespace internal {

// SimdMaskImpl ---------------------------------------------------------
/// item_size = 1 to store bits
template<int nb_items,int item_size,class Arch>
struct SimdMaskImpl;

// int, splittable version
template<int nb_items,int item_size,class Arch> requires ( item_size >= 8 && nb_items >= 2 )
struct SimdMaskImpl<nb_items,item_size,Arch> {
    static constexpr int split_size_0 = prev_pow_2( nb_items );
    static constexpr int split_size_1 = nb_items - split_size_0;
    static constexpr int splittable = 1;
    struct Split {
        SimdMaskImpl<split_size_0,item_size,Arch> v0;
        SimdMaskImpl<split_size_1,item_size,Arch> v1;
    };
  
    union {
        PI_<item_size>::T values[ nb_items ];
        Split split;
    } data;
};

// int, atomic version
template<int nb_items,int item_size,class Arch> requires ( item_size >= 8 && nb_items < 2 )
struct SimdMaskImpl<nb_items,item_size,Arch> {
    static constexpr int splittable = 0;
    union {
        PI_<item_size>::T values[ nb_items ];
    } data;
};

// bool, splittable version
template<int nb_items,int item_size,class Arch> requires ( item_size == 1 && nb_items >= 16 )
struct SimdMaskImpl<nb_items,item_size,Arch> {
    static constexpr int split_size_0 = prev_pow_2( nb_items );
    static constexpr int split_size_1 = nb_items - split_size_0;
    static constexpr int splittable = 1;
    struct Split {
        SimdMaskImpl<split_size_0,item_size,Arch> v0;
        SimdMaskImpl<split_size_1,item_size,Arch> v1;
    };
  
    union {
        BitVec<nb_items> values;
        Split split;
    } data;
};

// bool, atomic version
template<int nb_items,int item_size,class Arch> requires ( item_size == 1 && nb_items < 16 )
struct SimdMaskImpl<nb_items,item_size,Arch> {
    static constexpr int splittable = 0;
    union {
        BitVec<nb_items> values;
    } data;
};


/// Helper to make a SimdMaskImpl with a register. Version where mask values are stored in integer with size >= 8 bits
#define SIMD_MASK_IMPL_REG_LARGE( COND, NB_ITEMS, ITEM_SIZE, TREG ) \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) \
    struct SimdMaskImpl<NB_ITEMS,ITEM_SIZE,Arch> { \
        static constexpr int split_size_0 = prev_pow_2( NB_ITEMS ); \
        static constexpr int split_size_1 = NB_ITEMS - split_size_0; \
        struct Split { \
            SimdMaskImpl<NB_ITEMS/2,ITEM_SIZE,Arch> v0; \
            SimdMaskImpl<NB_ITEMS/2,ITEM_SIZE,Arch> v1; \
        }; \
        union { \
            PI##ITEM_SIZE values[ NB_ITEMS ]; \
            Split split; \
            TREG reg; \
        } data; \
    };

#define SIMD_MASK_IMPL_REG_BITS_UNSPLITABLE( COND, NB_ITEMS, TREG ) \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) \
    struct SimdMaskImpl<NB_ITEMS,1,Arch> { \
        union { \
            BitVec<NB_ITEMS> values; \
            TREG reg; \
        } data; \
    };

#define SIMD_MASK_IMPL_REG_BITS_SPLITABLE( COND, NB_ITEMS, TREG ) \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) \
    struct SimdMaskImpl<NB_ITEMS,1,Arch> { \
        static constexpr int split_size_0 = prev_pow_2( NB_ITEMS ); \
        static constexpr int split_size_1 = NB_ITEMS - split_size_0; \
        struct Split { \
            SimdMaskImpl<NB_ITEMS/2,1,Arch> v0; \
            SimdMaskImpl<NB_ITEMS/2,1,Arch> v1; \
        }; \
        union { \
            BitVec<NB_ITEMS> values; \
            Split split; \
            TREG reg; \
        } data; \
    };

// init_mask -----------------------------------------------------------------
template<int nb_items,int item_size,class Arch> HaD
void init_mask( SimdMaskImpl<nb_items,item_size,Arch> &mask, bool a, bool b, bool c, bool d, bool e, bool f, bool g, bool h ) {
    if constexpr ( item_size == 1 )
        mask.data.values.set_values( a, b, c, d, e, f, g, h );
    else
        mask.data.values = { a, b, c, d, e, f, g, h };
}

template<int nb_items,int item_size,class Arch> HaD
void init_mask( SimdMaskImpl<nb_items,item_size,Arch> &mask, bool a, bool b, bool c, bool d ) {
    if constexpr ( item_size == 1 )
        mask.data.values.set_values( a, b, c, d );
    else
        mask.data.values = { a, b, c, d };
}

template<int nb_items,int item_size,class Arch> HaD
void init_mask( SimdMaskImpl<nb_items,item_size,Arch> &mask, bool a, bool b ) {
    if constexpr ( item_size == 1 )
        mask.data.values.set_values( a, b );
    else
        mask.data.values = { a, b };
}

template<int nb_items,int item_size,class Arch> HaD
void init_mask( SimdMaskImpl<nb_items,item_size,Arch> &mask, bool a ) {
    if constexpr ( item_size == 1 )
        mask.data.values.set_value( a );
    else
        mask.data.values = { a };
}

// at ------------------------------------------------------------------------
template<int nb_items,int item_size,class Arch> HaD
bool at( const SimdMaskImpl<nb_items,item_size,Arch> &mask, int i ) {
    return mask.data.values[ i ] != 0;
}

// any, all ------------------------------------------------------------------
// template<int nb_items,int item_size,class Arch> HaD
// bool any( const SimdMaskImpl<nb_items,item_size,Arch> &mask )  {
//     return mask.data.values.any();
// }

template<int nb_items,class Arch> HaD
bool any( const SimdMaskImpl<nb_items,1,Arch> &mask )  {
    return mask.data.values.any();
}

template<int nb_items,int item_size,class Arch> HaD
bool all( const SimdMaskImpl<nb_items,item_size,Arch> &mask )  {
    return mask.data.values.all();
}

#define SIMD_MASK_IMPL_REG_REDUCTION( COND, NB_ITEMS, ITEM_SIZE, NAME, FUNC ) \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) HaD \
    bool NAME( const SimdMaskImpl<NB_ITEMS,ITEM_SIZE,Arch> &mask ) { \
        ASIMD_DEBUG_ON_OP(#NAME,#COND,#FUNC) return FUNC; \
    }


} // namespace internal
} // namespace asimd

