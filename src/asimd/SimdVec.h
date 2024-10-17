#pragma once

#include "internal/SimdVecImpl_Generic.h"
#include "internal/S.h"
#include "SimdMask.h"
#include "SimdSize.h"
#include "N.h"

#include "internal/SimdMaskImpl_X86.h"
#include "internal/SimdVecImpl_X86.h"
#include "Ptr.h"

namespace asimd {

/**
  Simd vector.
*/
template<class T_,int size_=SimdSize<T_,NativeCpu>::value,class Arch=NativeCpu>
struct SimdVec {
    using                                        Impl                  = SimdVecImpl::Impl<T_,size_,Arch>;
    using                                        T                     = T_;

    HaD                                          SimdVec               ( T a, T b, T c, T d, T e, T f, T g, T h ) { init_sc( impl, a, b, c, d, e, f, g, h ); }
    HaD                                          SimdVec               ( T a, T b, T c, T d, T e ) { init_sc( impl, a, b, c, d, e ); }
    HaD                                          SimdVec               ( T a, T b, T c, T d ) { init_sc( impl, a, b, c, d ); }
    HaD                                          SimdVec               ( T a, T b ) { init_sc( impl, a, b ); }
    HaD                                          SimdVec               ( T a ) { init_sc( impl, a ); }
    HaD                                          SimdVec               ( Impl impl ) : impl( impl ) {}
    HaD                                          SimdVec               () {}

    static HaD constexpr int                     size                  () { return size_; }

    // static ctors
    static HaD SimdVec                           iota                  ( T beg, T mul ) { return SimdVecImpl::iota( beg, mul, S<Impl>() ); }
    static HaD SimdVec                           iota                  ( T beg = 0 ) { return SimdVecImpl::iota( beg, S<Impl>() ); }

    //
    static void                                  prefetch              ( const T *beg ) { SimdVecImpl::prefetch( beg, N<sizeof(Impl)>(), S<Arch>() ); }

    // static versions of load/store
    template<class G> static HaD SimdVec         load_unaligned        ( const G *ptr ) { return SimdVecImpl::load_unaligned( ptr, S<Impl>() ); }
    template<class G> static HaD SimdVec         load_aligned          ( const G *ptr ) { return SimdVecImpl::load_aligned( ptr, S<Impl>() ); }
    template<class P> static HaD SimdVec         load                  ( const P &ptr ) { return SimdVecImpl::load( _chk_ptr( ptr ), S<Impl>() ); } ///< P::alignment, P::offset, data.get()

    template<class G> static HaD SimdVec         load_unaligned_stream ( const G *ptr ) { return SimdVecImpl::load_unaligned( ptr, S<Impl>() ); }
    template<class G> static HaD SimdVec         load_aligned_stream   ( const G *ptr ) { return SimdVecImpl::load_aligned_stream( ptr, S<Impl>() ); }
    template<class P> static HaD SimdVec         load_stream           ( const P &ptr ) { return SimdVecImpl::load_stream( _chk_ptr( ptr ), S<Impl>() ); } ///< P::alignment, P::offset, data.get()

    static HaD void                              init_unaligned        ( T *ptr, const SimdVec &vec ) { SimdVecImpl::init_unaligned( ptr, vec.impl ); }
    static HaD void                              init_aligned          ( T *ptr, const SimdVec &vec ) { SimdVecImpl::init_aligned( ptr, vec.impl ); }
    template<class P> static HaD void            init                  ( const P &ptr, const SimdVec &vec ) { SimdVecImpl::init( _chk_ptr( ptr ), vec.impl ); }

    static HaD void                              init_unaligned_stream ( T *ptr, const SimdVec &vec ) { SimdVecImpl::init_unaligned( ptr, vec.impl ); }
    static HaD void                              init_aligned_stream   ( T *ptr, const SimdVec &vec ) { SimdVecImpl::init_aligned_stream( ptr, vec.impl ); }
    template<class P> static HaD void            init_stream           ( const P &ptr, const SimdVec &vec ) { SimdVecImpl::init_stream( _chk_ptr( ptr ), vec.impl ); }

    static HaD void                              store_unaligned       ( T *ptr, const SimdVec &vec ) { SimdVecImpl::store_unaligned( ptr, vec.impl ); }
    static HaD void                              store_aligned         ( T *ptr, const SimdVec &vec ) { SimdVecImpl::store_aligned( ptr, vec.impl ); }
    template<class P> static HaD void            store                 ( const P &ptr, const SimdVec &vec ) { SimdVecImpl::store( _chk_ptr( ptr ), vec.impl ); }

    static HaD void                              store_unaligned_stream( T *ptr, const SimdVec &vec ) { SimdVecImpl::store_unaligned( ptr, vec.impl ); }
    static HaD void                              store_aligned_stream  ( T *ptr, const SimdVec &vec ) { SimdVecImpl::store_aligned_stream( ptr, vec.impl ); }
    template<class P> static HaD void            store_stream          ( const P &ptr, const SimdVec &vec ) { SimdVecImpl::store_stream( _chk_ptr( ptr ), vec.impl ); }

    // dynamic versions of load/store
    HaD void                                     store_unaligned       ( T *ptr ) const { store_unaligned( ptr, *this ); }
    HaD void                                     store_aligned         ( T *ptr ) const { store_aligned( ptr, *this ); }
    template<class P> HaD void                   store                 ( const P &ptr ) const { _chk_ptr( ptr ); store( ptr, *this ); }

    HaD void                                     store_unaligned_stream( T *ptr ) const { store_unaligned_stream( ptr, *this ); }
    HaD void                                     store_aligned_stream  ( T *ptr ) const { store_aligned_stream( ptr, *this ); }
    template<class P> HaD void                   store_stream          ( const P &ptr ) const { _chk_ptr( ptr ); store_stream( ptr, *this ); }

    HaD void                                     init_unaligned        ( T *ptr ) const { init_unaligned( ptr, *this ); }
    HaD void                                     init_aligned          ( T *ptr ) const { init_aligned( ptr, *this ); }
    template<class P> HaD void                   init                  ( const P &ptr ) const { init( ptr, *this ); }

    HaD void                                     init_unaligned_stream ( T *ptr ) const { init_unaligned( ptr, *this ); }
    HaD void                                     init_aligned_stream   ( T *ptr ) const { init_aligned_stream( ptr, *this ); }
    template<class P> HaD void                   init_stream           ( const P &ptr ) const { init_stream( ptr, *this ); }

    // scatter/gather
    template<class G,class V> static HaD void    scatter               ( G *ptr, const V &ind, const SimdVec &vec ) { SimdVecImpl::scatter( ptr, ind.impl, vec.impl ); }
    template<class G,class V> static HaD SimdVec gather                ( const G *data, const V &ind ) { return SimdVecImpl::gather( data, ind.impl, S<Impl>() ); }

    // selection
    HaD const T&                                 operator[]            ( int i ) const { return SimdVecImpl::at( impl, i ); }
    HaD T&                                       operator[]            ( int i ) { return SimdVecImpl::at( impl, i ); }
    HaD auto                                     sub_vec               ( N<size_> ) const { return *this; }
    HaD auto&                                    sub_vec               ( N<size_> ) { return *this; }
    template<int s> HaD auto                     sub_vec               ( N<s> ) const { return SimdVec<T,size_/2,Arch>( impl.data.split.v0 ).sub_vec( N<s>() ); }
    template<int s> HaD auto&                    sub_vec               ( N<s> ) { return reinterpret_cast<SimdVec<T,size_/2,Arch> &>( impl.data.split.v0 ).sub_vec( N<s>() ); }
    HaD const T*                                 begin                 () const { return &operator[]( 0 ); }
    HaD const T*                                 end                   () const { return begin() + size(); }

    // arithmetic operators
    HaD SimdVec                                  operator<<            ( const SimdVec &that ) const { return SimdVecImpl::sll( impl, that.impl ); }
    HaD SimdVec                                  operator&             ( const SimdVec &that ) const { return SimdVecImpl::anb( impl, that.impl ); }
    HaD SimdVec                                  operator+             ( const SimdVec &that ) const { return SimdVecImpl::add( impl, that.impl ); }
    HaD SimdVec                                  operator-             ( const SimdVec &that ) const { return SimdVecImpl::sub( impl, that.impl ); }
    HaD SimdVec                                  operator*             ( const SimdVec &that ) const { return SimdVecImpl::mul( impl, that.impl ); }
    HaD SimdVec                                  operator/             ( const SimdVec &that ) const { return SimdVecImpl::div( impl, that.impl ); }

    // comparison: return an Op_... that can be converted to a SimdMask or a SimdVec
    HaD auto                                     operator>             ( const SimdVec &that ) const { return SimdVecImpl::gt ( impl, that.impl );  }
    HaD auto                                     operator<             ( const SimdVec &that ) const { return SimdVecImpl::lt ( impl, that.impl );  }

    // self arithmetic operators
    HaD SimdVec&                                 operator+=            ( const auto &that ) { *this = *this + that; return *this; }
    HaD SimdVec&                                 operator-=            ( const auto &that ) { *this = *this - that; return *this; }
    HaD SimdVec&                                 operator*=            ( const auto &that ) { *this = *this * that; return *this; }
    HaD SimdVec&                                 operator/=            ( const auto &that ) { *this = *this / that; return *this; }

    HaD T                                        sum                   () const { return SimdVecImpl::horizontal_sum( impl ); }

    template<class P> static const P&            _chk_ptr              ( const P &ptr ) { static_assert( P::alignment && ( P::offset != P::alignment ), "this method is expecting a asimd::Ptr<> like object (with `alignment`, `offset` static attributes and a `get` method)" ); return ptr; }

    Impl                                         impl;                 ///<
};

template<class T,int size,class Arch> HaD
SimdVec<T,size,Arch> min( const SimdVec<T,size,Arch> &a, const SimdVec<T,size,Arch> &b ) { return SimdVecImpl::min( a.impl, b.impl ); }

template<class T,int size,class Arch> HaD
SimdVec<T,size,Arch> max( const SimdVec<T,size,Arch> &a, const SimdVec<T,size,Arch> &b ) { return SimdVecImpl::max( a.impl, b.impl ); }

} // namespace asimd
