#pragma once

#include "impl/SimdMaskImpl_Generic.h"
#include "impl/SimdMaskImpl_X86.h"

#include "architectures/NativeCpu.h"
#include "support/HaD.h"

namespace asimd {

/**
  Simd mask.

  item_size in bits
*/
template<int nb_items,int item_size=1,class Arch=NativeCpu>
struct SimdMask {
    using                     Impl      = internal::SimdMaskImpl<nb_items,item_size,Arch>;

    HaD                       SimdMask  ( bool a, bool b, bool c, bool d, bool e, bool f, bool g, bool h ) { internal::init_mask( impl, a, b, c, d, e, f, g, h ); }
    HaD                       SimdMask  ( bool a, bool b, bool c, bool d, bool e ) { internal::init_mask( impl, a, b, c, d, e ); }
    HaD                       SimdMask  ( bool a, bool b, bool c, bool d ) { internal::init_mask( impl, a, b, c, d ); }
    HaD                       SimdMask  ( bool a, bool b ) { internal::init_mask( impl, a, b ); }
    HaD                       SimdMask  ( bool a ) { internal::init_mask( impl, a ); }
    HaD                       SimdMask  ( Impl impl ) : impl( impl ) {}
    HaD                       SimdMask  () {}

    // individual items
    HaD bool                  operator[]( int i ) const { return internal::at( impl, i ); }
    static HaD constexpr int  size      () { return nb_items; }

    // arithmetic operators
    // HaD SimdMask           operator& ( const SimdMask &that ) const { return SimdVecinternal::anb( impl, that.impl ); }

    Impl                      impl;     ///<
};

template<int nb_items,int item_size,class Arch>
SimdMask<nb_items,item_size,Arch> simd_mask_from_simd_mask_impl( internal::SimdMaskImpl<nb_items,item_size,Arch> &&impl ) { return impl; }

template<int nb_items,int item_size,class Arch> HaD
bool any( const SimdMask<nb_items,item_size,Arch> &a ) { return internal::any( a.impl ); }

template<int nb_items,int item_size,class Arch> HaD
bool all( const SimdMask<nb_items,item_size,Arch> &a ) { return internal::all( a.impl ); }

} // namespace asimd

