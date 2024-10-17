#pragma once

#include "impl/SimdVecImpl_Generic.h"
#include "impl/SimdVecImpl_X86.h" // IWYU pragma: export

#include "SimdMask.h"
#include "SimdSize.h"
//#include "Ptr.h"

namespace asimd {

/**
  Simd vector.
*/
template<class T_,int size_=SimdSize<T_,NativeCpu>::value,class Arch=NativeCpu>
struct SimdVec {
    using                                        Impl                  = internal::SimdVecImpl<T_,size_,Arch>;
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
    static HaD SimdVec                           iota                  ( T beg, T mul ) { return internal::iota( beg, mul, S<Impl>() ); }
    static HaD SimdVec                           iota                  ( T beg = 0 ) { return internal::iota( beg, S<Impl>() ); }

    //
    static void                                  prefetch              ( const T *beg ) { internal::prefetch( beg, N<sizeof(Impl)>(), S<Arch>() ); }

    // static versions of load/store
    template<class G> static HaD SimdVec         load_unaligned        ( const G *ptr ) { return internal::load_unaligned( ptr, S<Impl>() ); }
    template<class G> static HaD SimdVec         load_aligned          ( const G *ptr ) { return internal::load_aligned( ptr, S<Impl>() ); }
    template<class P> static HaD SimdVec         load                  ( const P &ptr ) { return internal::load( _chk_ptr( ptr ), S<Impl>() ); } ///< P::alignment, P::offset, data.get()

    template<class G> static HaD SimdVec         load_unaligned_stream ( const G *ptr ) { return internal::load_unaligned( ptr, S<Impl>() ); }
    template<class G> static HaD SimdVec         load_aligned_stream   ( const G *ptr ) { return internal::load_aligned_stream( ptr, S<Impl>() ); }
    template<class P> static HaD SimdVec         load_stream           ( const P &ptr ) { return internal::load_stream( _chk_ptr( ptr ), S<Impl>() ); } ///< P::alignment, P::offset, data.get()

    static HaD void                              init_unaligned        ( T *ptr, const SimdVec &vec ) { internal::init_unaligned( ptr, vec.impl ); }
    static HaD void                              init_aligned          ( T *ptr, const SimdVec &vec ) { internal::init_aligned( ptr, vec.impl ); }
    template<class P> static HaD void            init                  ( const P &ptr, const SimdVec &vec ) { internal::init( _chk_ptr( ptr ), vec.impl ); }

    static HaD void                              init_unaligned_stream ( T *ptr, const SimdVec &vec ) { internal::init_unaligned( ptr, vec.impl ); }
    static HaD void                              init_aligned_stream   ( T *ptr, const SimdVec &vec ) { internal::init_aligned_stream( ptr, vec.impl ); }
    template<class P> static HaD void            init_stream           ( const P &ptr, const SimdVec &vec ) { internal::init_stream( _chk_ptr( ptr ), vec.impl ); }

    static HaD void                              store_unaligned       ( T *ptr, const SimdVec &vec ) { internal::store_unaligned( ptr, vec.impl ); }
    static HaD void                              store_aligned         ( T *ptr, const SimdVec &vec ) { internal::store_aligned( ptr, vec.impl ); }
    template<class P> static HaD void            store                 ( const P &ptr, const SimdVec &vec ) { internal::store( _chk_ptr( ptr ), vec.impl ); }

    static HaD void                              store_unaligned_stream( T *ptr, const SimdVec &vec ) { internal::store_unaligned( ptr, vec.impl ); }
    static HaD void                              store_aligned_stream  ( T *ptr, const SimdVec &vec ) { internal::store_aligned_stream( ptr, vec.impl ); }
    template<class P> static HaD void            store_stream          ( const P &ptr, const SimdVec &vec ) { internal::store_stream( _chk_ptr( ptr ), vec.impl ); }

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
    template<class G,class V> static HaD void    scatter               ( G *ptr, const V &ind, const SimdVec &vec ) { internal::scatter( ptr, ind.impl, vec.impl ); }
    template<class G,class V> static HaD SimdVec gather                ( const G *data, const V &ind ) { return internal::gather( data, ind.impl, S<Impl>() ); }

    // selection
    HaD const T&                                 operator[]            ( int i ) const { return internal::at( impl, i ); }
    HaD T&                                       operator[]            ( int i ) { return internal::at( impl, i ); }
    HaD auto                                     sub_vec               ( N<size_> ) const { return *this; }
    HaD auto&                                    sub_vec               ( N<size_> ) { return *this; }
    template<int s> HaD auto                     sub_vec               ( N<s> ) const { return SimdVec<T,size_/2,Arch>( impl.data.split.v0 ).sub_vec( N<s>() ); }
    template<int s> HaD auto&                    sub_vec               ( N<s> ) { return reinterpret_cast<SimdVec<T,size_/2,Arch> &>( impl.data.split.v0 ).sub_vec( N<s>() ); }
    HaD const T*                                 begin                 () const { return &operator[]( 0 ); }
    HaD const T*                                 end                   () const { return begin() + size(); }

    // arithmetic operators
    HaD SimdVec                                  operator<<            ( const SimdVec &that ) const { return internal::sll( impl, that.impl ); }
    HaD SimdVec                                  operator&             ( const SimdVec &that ) const { return internal::anb( impl, that.impl ); }
    HaD SimdVec                                  operator+             ( const SimdVec &that ) const { return internal::add( impl, that.impl ); }
    HaD SimdVec                                  operator-             ( const SimdVec &that ) const { return internal::sub( impl, that.impl ); }
    HaD SimdVec                                  operator*             ( const SimdVec &that ) const { return internal::mul( impl, that.impl ); }
    HaD SimdVec                                  operator/             ( const SimdVec &that ) const { return internal::div( impl, that.impl ); }

    // comparison: return an Op_... that can be converted to a SimdMask or a SimdVec
    HaD auto                                     operator>             ( const SimdVec &that ) const { return internal::gt ( impl, that.impl );  }
    HaD auto                                     operator<             ( const SimdVec &that ) const { return internal::lt ( impl, that.impl );  }

    // self arithmetic operators
    HaD SimdVec&                                 operator+=            ( const auto &that ) { *this = *this + that; return *this; }
    HaD SimdVec&                                 operator-=            ( const auto &that ) { *this = *this - that; return *this; }
    HaD SimdVec&                                 operator*=            ( const auto &that ) { *this = *this * that; return *this; }
    HaD SimdVec&                                 operator/=            ( const auto &that ) { *this = *this / that; return *this; }

    HaD T                                        sum                   () const { return internal::horizontal_sum( impl ); }

    template<class P> static const P&            _chk_ptr              ( const P &ptr ) { static_assert( P::alignment && ( P::offset != P::alignment ), "this method is expecting a asimd::Ptr<> like object (with `alignment`, `offset` static attributes and a `get` method)" ); return ptr; }

    Impl                                         impl;                 ///<
};

#define SIMD_VEC_IMPL_CMP_OP( NAME, OP ) \
    template<class T,int size,class Arch> auto as_a_simd_mask( const internal::Op_##NAME<T,size,Arch> &op ) { return simd_mask_from_simd_mask_impl( internal::NAME##_as_a_simd_mask( op.a, op.b ) ); } \
    template<class T,int size,class Arch> auto as_a_simd_vec( const internal::Op_##NAME<T,size,Arch> &op ) { using P = PI_<8*sizeof(T)>::T; return SimdVec<P,size,Arch>( internal::NAME##_as_a_simd_vec( op.a, op.b, S<internal::SimdVecImpl<P,size,Arch>>() ) ); } \
    template<class T,int size,class Arch> bool any( const internal::Op_##NAME<T,size,Arch> &op ) { return any( as_a_simd_mask( op ) ); } \
    template<class T,int size,class Arch> bool all( const internal::Op_##NAME<T,size,Arch> &op ) { return all( as_a_simd_mask( op ) ); } \

SIMD_VEC_IMPL_CMP_OP( lt, < )
SIMD_VEC_IMPL_CMP_OP( gt, > )

#undef SIMD_VEC_IMPL_CMP_OP

template<class T,int size,class Arch> HaD
SimdVec<T,size,Arch> min( const SimdVec<T,size,Arch> &a, const SimdVec<T,size,Arch> &b ) { return internal::min( a.impl, b.impl ); }

template<class T,int size,class Arch> HaD
SimdVec<T,size,Arch> max( const SimdVec<T,size,Arch> &a, const SimdVec<T,size,Arch> &b ) { return internal::max( a.impl, b.impl ); }

} // namespace asimd
