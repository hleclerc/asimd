#pragma once

#include "internal/SimdMaskImpl_Generic.h"
// #include "internal/SimdVecImpl_X86.h"
#include "architectures/NativeCpu.h"
// #include "internal/S.h"
#include "N.h"

namespace asimd {

/**
  Simd vector.
*/
template<int size_,class Arch=NativeCpu>
struct SimdMask {
    using                                        Impl                  = SimdMaskImpl::Impl<size_,Arch>;

    // HaD                                       SimdVec               ( T a, T b, T c, T d, T e, T f, T g, T h ) { init_sc( impl, a, b, c, d, e, f, g, h ); }
    // HaD                                       SimdVec               ( T a, T b, T c, T d, T e ) { init_sc( impl, a, b, c, d, e ); }
    // HaD                                       SimdVec               ( T a, T b, T c, T d ) { init_sc( impl, a, b, c, d ); }
    // HaD                                       SimdVec               ( T a, T b ) { init_sc( impl, a, b ); }
    // HaD                                       SimdVec               ( T a ) { init_sc( impl, a ); }
    HaD                                          SimdMask              ( Impl impl ) : impl( impl ) {}
    HaD                                          SimdMask              () {}


    // selection
    HaD bool                                     operator[]            ( int i ) const { return SimdMaskImpl::at( impl, i ); }
    static HaD constexpr int                     size                  () { return size_; }

    HaD auto                                     sub_vec               ( N<size_> ) const { return *this; }
    HaD auto&                                    sub_vec               ( N<size_> ) { return *this; }
    template<int s> HaD auto                     sub_vec               ( N<s> ) const { return SimdMask<size_/2,Arch>( impl.data.split.v0 ).sub_vec( N<s>() ); }
    template<int s> HaD auto&                    sub_vec               ( N<s> ) { return reinterpret_cast<SimdMask<size_/2,Arch> &>( impl.data.split.v0 ).sub_vec( N<s>() ); }

    // // arithmetic operators
    // HaD SimdVec                               operator<<            ( const SimdVec &that ) const { return SimdVecImpl::sll( impl, that.impl ); }
    // HaD SimdVec                               operator&             ( const SimdVec &that ) const { return SimdVecImpl::anb( impl, that.impl ); }
    // HaD SimdVec                               operator+             ( const SimdVec &that ) const { return SimdVecImpl::add( impl, that.impl ); }
    // HaD SimdVec                               operator-             ( const SimdVec &that ) const { return SimdVecImpl::sub( impl, that.impl ); }
    // HaD SimdVec                               operator*             ( const SimdVec &that ) const { return SimdVecImpl::mul( impl, that.impl ); }
    // HaD SimdVec                               operator/             ( const SimdVec &that ) const { return SimdVecImpl::div( impl, that.impl ); }

    // HaD auto                                  operator>             ( const SimdVec &that ) const { return SimdVecImpl::gt ( impl, that.impl );  }
    // HaD auto                                  operator<             ( const SimdVec &that ) const { return SimdVecImpl::lt ( impl, that.impl );  }

    // // self arithmetic operators
    // HaD SimdVec&                              operator+=            ( const auto &that ) { *this = *this + that; return *this; }
    // HaD SimdVec&                              operator-=            ( const auto &that ) { *this = *this - that; return *this; }
    // HaD SimdVec&                              operator*=            ( const auto &that ) { *this = *this * that; return *this; }
    // HaD SimdVec&                              operator/=            ( const auto &that ) { *this = *this / that; return *this; }

    // HaD T                                     sum                   () const { return SimdVecImpl::horizontal_sum( impl ); }

    Impl                                         impl;                 ///<
};

// template<class T,int size,class Arch> HaD
// SimdVec<T,size,Arch> max( const SimdVec<T,size,Arch> &a, const SimdVec<T,size,Arch> &b ) { return SimdVecImpl::max( a.impl, b.impl ); }

} // namespace asimd
