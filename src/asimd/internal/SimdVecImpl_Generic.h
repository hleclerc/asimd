#pragma once

#include "../architectures/NativeCpu.h"
#include "HaD.h"
#include "S.h"

#include <ostream>

#ifndef ASIMD_DEBUG_ON_OP
#define ASIMD_DEBUG_ON_OP( NAME, COND )
#endif // ASIMD_DEBUG_ON_OP

namespace asimd {
namespace SimdVecImpl {

template<class I> inline constexpr
I prew_pow_2( I v ) {
    return 1 << ( sizeof( I ) * 8 - __builtin_clz( v - 1 ) - 1 );
}

// Impl ---------------------------------------------------------
template<class T,int size,class Arch,class Enable=void>
struct Impl {
    static constexpr int split_size_0 = prew_pow_2( size );
    static constexpr int split_size_1 = size - split_size_0;
    struct Split {
        Impl<T,split_size_0,Arch> v0;
        Impl<T,split_size_1,Arch> v1;
    };
    union {
        T     values[ size ];
        Split split;
    } data;
};

template<class T,class Arch,class Enable>
struct Impl<T,1,Arch,Enable> {
    union {
        T values[ 1 ];
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

// BitSet --------------------------------------------------------
//template<int size_,class Arch>
//struct BitSet {
//    enum {            size           = size_ };

//    /**/              BitSet         ( const BitSet<size/2,Arch> &a, const BitSet<size/2,Arch> &b ) : data( cat( a.data, b.data ) ) {}
//    template<class I> BitSet         ( I v ) : data( v ) {}

//    void              write_to_stream( std::ostream &os ) const { for( int i = 0; i < size; ++i ) os << data[ i ]; }
//    bool              operator[]     ( int i ) const { return data[ i ]; }
//    auto              operator[]     ( int i ) { return data[ i ]; }

//    std::bitset<size> data;
//};


// at ------------------------------------------------------------------------
template<class T,int size,class Arch> HaD
const T &at( const Impl<T,size,Arch> &vec, int i ) {
    return vec.data.values[ i ];
}

template<class T,int size,class Arch> HaD
T &at( Impl<T,size,Arch> &vec, int i ) {
    return vec.data.values[ i ];
}

// init ----------------------------------------------------------------------
template<class T,int size,class Arch,class G> HaD
void init_sc( Impl<T,size,Arch> &vec, G a, G b, G c, G d, G e, G f, G g, G h ) {
    vec.data.values[ 0 ] = a;
    vec.data.values[ 1 ] = b;
    vec.data.values[ 2 ] = c;
    vec.data.values[ 3 ] = d;
    vec.data.values[ 4 ] = e;
    vec.data.values[ 5 ] = f;
    vec.data.values[ 6 ] = g;
    vec.data.values[ 7 ] = h;
}

template<class T,int size,class Arch,class G> HaD
void init_sc( Impl<T,size,Arch> &vec, G a, G b, G c, G d, G e ) {
    vec.data.values[ 0 ] = a;
    vec.data.values[ 1 ] = b;
    vec.data.values[ 2 ] = c;
    vec.data.values[ 3 ] = d;
    vec.data.values[ 4 ] = e;
}

template<class T,int size,class Arch,class G> HaD
void init_sc( Impl<T,size,Arch> &vec, G a, G b, G c, G d ) {
    vec.data.values[ 0 ] = a;
    vec.data.values[ 1 ] = b;
    vec.data.values[ 2 ] = c;
    vec.data.values[ 3 ] = d;
}

template<class T,int size,class Arch,class G> HaD
void init_sc( Impl<T,size,Arch> &vec, G a, G b ) {
    vec.data.values[ 0 ] = a;
    vec.data.values[ 1 ] = b;
}

template<class T,int size,class Arch,class G> HaD
void init_sc( Impl<T,size,Arch> &vec, G a ) {
    for( int i = 0; i < size; ++i )
        vec.data.values[ i ] = a;
}

template<class T,int size,int part,class Arch> HaD
void init_sc( Impl<T,size,Arch> &vec, Impl<T,part,Arch> a, Impl<T,size-part,Arch> b ) {
    vec.data.split.v0 = a;
    vec.data.split.v1 = b;
}

#define SIMD_VEC_IMPL_REG_INIT_1( COND, T, SIZE, FUNC ) \
    template<class Arch> HaD \
    typename std::enable_if<Arch::template Has<features::COND>::value>::type init_sc( Impl<T,SIZE,Arch> &vec, T a ) { \
        vec.data.reg = FUNC; \
    }

#define SIMD_VEC_IMPL_REG_INIT_2( COND, T, SIZE, FUNC ) \
    template<class Arch> HaD \
    typename std::enable_if<Arch::template Has<features::COND>::value>::type init_sc( Impl<T,SIZE,Arch> &vec, T a, T b ) { \
        vec.data.reg = FUNC; \
    }

#define SIMD_VEC_IMPL_REG_INIT_4( COND, T, SIZE, FUNC ) \
    template<class Arch> HaD \
    typename std::enable_if<Arch::template Has<features::COND>::value>::type init_sc( Impl<T,SIZE,Arch> &vec, T a, T b, T c, T d ) { \
        vec.data.reg = FUNC; \
    }

#define SIMD_VEC_IMPL_REG_INIT_8( COND, T, SIZE, FUNC ) \
    template<class Arch> HaD \
    typename std::enable_if<Arch::template Has<features::COND>::value>::type init_sc( Impl<T,SIZE,Arch> &vec, T a, T b, T c, T d, T e, T f, T g, T h ) { \
        vec.data.reg = FUNC; \
    }

#define SIMD_VEC_IMPL_REG_INIT_16( COND, T, SIZE, FUNC ) \
    template<class Arch> HaD \
    typename std::enable_if<Arch::template Has<features::COND>::value>::type init_sc( Impl<T,SIZE,Arch> &vec, T a, T b, T c, T d, T e, T f, T g, T h, T i, T j, T k, T l, T m, T n, T o, T p ) { \
        vec.data.reg = FUNC; \
    }

// prefetch ----------------------------------------------------------------------
template<int len,class N_len,class S_Arch>
void prefetch( const void *, N_len, S_Arch ) {
}

// load_unaligned ----------------------------------------------------------------------------
template<class G,class T,int size,class Arch> HaD
Impl<T,size,Arch> load_unaligned( const G *data, S<Impl<T,size,Arch>> ) {
    Impl<T,size,Arch> res;
    res.data.split.v0 = load_unaligned( data                                  , S<Impl<T,Impl<T,size,Arch>::split_size_0,Arch>>() );
    res.data.split.v1 = load_unaligned( data + Impl<T,size,Arch>::split_size_0, S<Impl<T,Impl<T,size,Arch>::split_size_1,Arch>>() );
    return res;
}

template<class G,class T,class Arch> HaD
Impl<T,1,Arch> load_unaligned( const G *data, S<Impl<T,1,Arch>> ) {
    Impl<T,1,Arch> res;
    res.data.values[ 0 ] = *data;
    return res;
}

#define SIMD_VEC_IMPL_REG_LOAD_UNALIGNED( COND, T, SIZE, FUNC ) \
    template<class Arch> HaD \
    typename std::enable_if<Arch::template Has<features::COND>::value,Impl<T,SIZE,Arch>>::type load_unaligned( const T *data, S<Impl<T,SIZE,Arch>> ) { \
        ASIMD_DEBUG_ON_OP("load_unaligned",#COND) Impl<T,SIZE,Arch> res; res.data.reg = FUNC; return res; \
    }

#define SIMD_VEC_IMPL_REG_LOAD_FOT( COND, T, G, SIZE, FUNC ) \
    template<class Arch> HaD \
    typename std::enable_if<Arch::template Has<features::COND>::value,Impl<T,SIZE,Arch>>::type load_unaligned( const G *data, S<Impl<T,SIZE,Arch>> ) { \
        ASIMD_DEBUG_ON_OP("store_unaligned_fot",#COND) Impl<T,SIZE,Arch> res; res.data.reg = FUNC; return res; \
    }

// load_aligned ---------------------------------------------------------------------------------
template<class G,class T,int size,class Arch> HaD
Impl<T,size,Arch> load_aligned( const G *data, S<Impl<T,size,Arch>> ) {
    Impl<T,size,Arch> res;
    res.data.split.v0 = load_aligned( data                                  , S<Impl<T,Impl<T,size,Arch>::split_size_0,Arch>>() );
    res.data.split.v1 = load_aligned( data + Impl<T,size,Arch>::split_size_0, S<Impl<T,Impl<T,size,Arch>::split_size_1,Arch>>() );
    return res;
}

template<class G,class T,class Arch> HaD
Impl<T,1,Arch> load_aligned( const G *data, S<Impl<T,1,Arch>> ) {
    Impl<T,1,Arch> res;
    res.data.values[ 0 ] = *data;
    return res;
}

template<class G,class T,int size,class Arch> HaD
Impl<T,size,Arch> load_aligned_stream( const G *data, S<Impl<T,size,Arch>> ) {
    Impl<T,size,Arch> res;
    res.data.split.v0 = load_aligned_stream( data                                  , S<Impl<T,Impl<T,size,Arch>::split_size_0,Arch>>() );
    res.data.split.v1 = load_aligned_stream( data + Impl<T,size,Arch>::split_size_0, S<Impl<T,Impl<T,size,Arch>::split_size_1,Arch>>() );
    return res;
}

template<class G,class T,class Arch> HaD
Impl<T,1,Arch> load_aligned_stream( const G *data, S<Impl<T,1,Arch>> ) {
    Impl<T,1,Arch> res;
    res.data.values[ 0 ] = *data;
    return res;
}

template<class P,class T,int size,class Arch> HaD
Impl<T,size,Arch> load( const P &data, S<Impl<T,size,Arch>> ) {
    Impl<T,size,Arch> res;
    res.data.split.v0 = load( data                                       , S<Impl<T,Impl<T,size,Arch>::split_size_0,Arch>>() );
    res.data.split.v1 = load( data + N<Impl<T,size,Arch>::split_size_0>(), S<Impl<T,Impl<T,size,Arch>::split_size_1,Arch>>() );
    return res;
}

template<class P,class T,class Arch> HaD
Impl<T,1,Arch> load( const P &data, S<Impl<T,1,Arch>> ) {
    Impl<T,1,Arch> res;
    res.data.values[ 0 ] = *data.get();
    return res;
}

template<class P,class T,int size,class Arch> HaD
Impl<T,size,Arch> load_stream( const P &data, S<Impl<T,size,Arch>> ) {
    Impl<T,size,Arch> res;
    res.data.split.v0 = load_stream( data                                       , S<Impl<T,Impl<T,size,Arch>::split_size_0,Arch>>() );
    res.data.split.v1 = load_stream( data + N<Impl<T,size,Arch>::split_size_0>(), S<Impl<T,Impl<T,size,Arch>::split_size_1,Arch>>() );
    return res;
}

template<class P,class T,class Arch> HaD
Impl<T,1,Arch> load_stream( const P &data, S<Impl<T,1,Arch>> ) {
    Impl<T,1,Arch> res;
    res.data.values[ 0 ] = *data.get();
    return res;
}

#define SIMD_VEC_IMPL_REG_LOAD_ALIGNED( COND, T, SIZE, MIN_ALIG, FUNC ) \
    template<class Arch> HaD \
    auto load_aligned( const T *data, S<Impl<T,SIZE,Arch>> ) -> typename std::enable_if<Arch::template Has<features::COND>::value,Impl<T,SIZE,Arch>>::type { \
        ASIMD_DEBUG_ON_OP("load_aligned",#COND) Impl<T,SIZE,Arch> res; res.data.reg = FUNC; return res; \
    } \
    template<class P,class Arch> HaD \
    auto load( const P &data, S<Impl<T,SIZE,Arch>> s ) -> typename std::enable_if<std::is_same<T,typename std::decay<decltype(*data)>::type>::value && Arch::template Has<features::COND>::value,Impl<T,SIZE,Arch>>::type { \
        return P::alignment % MIN_ALIG || P::offset % MIN_ALIG ? load_unaligned( data.get(), s ) : load_aligned( data.get(), s ); \
    }

#define SIMD_VEC_IMPL_REG_LOAD_ALIGNED_STREAM( COND, T, SIZE, MIN_ALIG, FUNC ) \
    template<class Arch> HaD \
    auto load_aligned_stream( const T *data, S<Impl<T,SIZE,Arch>> ) -> typename std::enable_if<Arch::template Has<features::COND>::value,Impl<T,SIZE,Arch>>::type { \
        ASIMD_DEBUG_ON_OP("load_aligned_stream",#COND) Impl<T,SIZE,Arch> res; res.data.reg = FUNC; return res; \
    } \
    template<class P,class Arch> HaD \
    auto load_stream( const P &data, S<Impl<T,SIZE,Arch>> s ) -> typename std::enable_if<std::is_same<T,typename std::decay<decltype(*data)>::type>::value && Arch::template Has<features::COND>::value,Impl<T,SIZE,Arch>>::type { \
        return P::alignment % MIN_ALIG || P::offset % MIN_ALIG ? load_unaligned( data.get(), s ) : load_aligned_stream( data.get(), s ); \
    }

#define SIMD_VEC_IMPL_REG_LOAD_ALIGNED_FOT( COND, T, G, SIZE, MIN_ALIG, FUNC ) \
    template<class Arch> HaD \
    auto load_aligned( const G *data, S<Impl<T,SIZE,Arch>> ) -> typename std::enable_if<Arch::template Has<features::COND>::value,Impl<T,SIZE,Arch>>::type { \
        ASIMD_DEBUG_ON_OP("load_aligned_fot",#COND) Impl<T,SIZE,Arch> res; res.data.reg = FUNC; return res; \
    } \
    template<class P,int a> HaD \
    auto load( const P &data, S<Impl<T,SIZE,Arch>> s ) -> typename std::enable_if<std::is_same<G,typename std::decay<decltype(*data)>::type>::value && Arch::template Has<features::COND>::value,Impl<T,SIZE,Arch>>::type { \
        return P::alignment % MIN_ALIG || P::offset % MIN_ALIG ? load_unaligned( data.get(), s ) : load_aligned( data.get(), s ); \
    }

// store and init unaligned -----------------------------------------------------------------------
template<class G,class T,int size,class Arch> HaD
void store_unaligned( G *data, const Impl<T,size,Arch> &impl ) {
    store_unaligned( data                    , impl.data.split.v0 );
    store_unaligned( data + impl.split_size_0, impl.data.split.v1 );
}

template<class G,class T,class Arch> HaD
void store_unaligned( G *data, const Impl<T,1,Arch> &impl ) {
    *data = impl.data.values[ 0 ];
}

template<class G,class T,int size,class Arch> HaD
void init_unaligned( G *data, const Impl<T,size,Arch> &impl ) {
    init_unaligned( data                    , impl.data.split.v0 );
    init_unaligned( data + impl.split_size_0, impl.data.split.v1 );
}

template<class G,class T,class Arch> HaD
void init_unaligned( G *data, const Impl<T,1,Arch> &impl ) {
    new ( data ) G( impl.data.values[ 0 ] );
}

#define SIMD_VEC_IMPL_REG_STORE_UNALIGNED( COND, T, SIZE, FUNC ) \
    template<class Arch> HaD \
    auto store_unaligned( T *data, const Impl<T,SIZE,Arch> &impl ) -> typename std::enable_if<Arch::template Has<features::COND>::value>::type { \
        ASIMD_DEBUG_ON_OP("store_unaligned",#COND) FUNC; \
    } \
    template<class Arch> HaD \
    auto init_unaligned( T *data, const Impl<T,SIZE,Arch> &impl ) -> typename std::enable_if<Arch::template Has<features::COND>::value>::type { \
        ASIMD_DEBUG_ON_OP("init_unaligned",#COND) FUNC; \
    }

// store and init aligned -----------------------------------------------------------------------
template<class G,class T,int size,class Arch> HaD
void store_aligned( G *data, const Impl<T,size,Arch> &impl ) {
    store_aligned( data                    , impl.data.split.v0 );
    store_aligned( data + impl.split_size_0, impl.data.split.v1 );
}

template<class G,class T,class Arch> HaD
void store_aligned( G *data, const Impl<T,1,Arch> &impl ) {
    *data = impl.data.values[ 0 ];
}

template<class P,class T,int size,class Arch> HaD
void store( const P &data, const Impl<T,size,Arch> &impl ) {
    store( data                         , impl.data.split.v0 );
    store( data + N<impl.split_size_0>(), impl.data.split.v1 );
}

template<class P,class T,class Arch> HaD
void store( const P &data, const Impl<T,1,Arch> &impl ) {
    *data = impl.data.values[ 0 ];
}

template<class G,class T,int size,class Arch> HaD
void store_aligned_stream( G *data, const Impl<T,size,Arch> &impl ) {
    store_aligned_stream( data                    , impl.data.split.v0 );
    store_aligned_stream( data + impl.split_size_0, impl.data.split.v1 );
}

template<class G,class T,class Arch> HaD
void store_aligned_stream( G *data, const Impl<T,1,Arch> &impl ) {
    *data = impl.data.values[ 0 ];
}

template<class P,class T,int size,class Arch> HaD
void store_stream( const P &data, const Impl<T,size,Arch> &impl ) {
    store_stream( data                                       , impl.data.split.v0 );
    store_stream( data + N<Impl<T,size,Arch>::split_size_0>(), impl.data.split.v1 );
}

template<class P,class T,class Arch> HaD
void store_stream( const P &data, const Impl<T,1,Arch> &impl ) {
    *data = impl.data.values[ 0 ];
}

template<class G,class T,int size,class Arch> HaD
void init_aligned( G *data, const Impl<T,size,Arch> &impl ) {
    init_aligned( data                    , impl.data.split.v0 );
    init_aligned( data + impl.split_size_0, impl.data.split.v1 );
}

template<class G,class T,class Arch> HaD
void init_aligned( G *data, const Impl<T,1,Arch> &impl ) {
    new ( data ) G( impl.data.values[ 0 ] );
}

template<class P,class T,int size,class Arch> HaD
void init( const P &data, const Impl<T,size,Arch> &impl ) {
    init( data                         , impl.data.split.v0 );
    init( data + N<impl.split_size_0>(), impl.data.split.v1 );
}

template<class P,class T,class Arch> HaD
void init( const P &data, const Impl<T,1,Arch> &impl ) {
    using G = typename std::decay<decltype(*data)>::type;
    new ( data.get() ) G( impl.data.values[ 0 ] );
}

template<class G,class T,int size,class Arch> HaD
void init_aligned_stream( G *data, const Impl<T,size,Arch> &impl ) {
    init_aligned_stream( data                    , impl.data.split.v0 );
    init_aligned_stream( data + impl.split_size_0, impl.data.split.v1 );
}

template<class G,class T,class Arch> HaD
void init_aligned_stream( G *data, const Impl<T,1,Arch> &impl ) {
    new ( data ) G( impl.data.values[ 0 ] );
}

template<class P,class T,int size,class Arch> HaD
void init_stream( const P &data, const Impl<T,size,Arch> &impl ) {
    init_stream( data                         , impl.data.split.v0 );
    init_stream( data + N<impl.split_size_0>(), impl.data.split.v1 );
}

template<class P,class T,class Arch> HaD
void init_stream( const P &data, const Impl<T,1,Arch> &impl ) {
    using G = typename std::decay<decltype(*data)>::type;
    new ( data.get() ) G( impl.data.values[ 0 ] );
}

#define SIMD_VEC_IMPL_REG_STORE_ALIGNED( COND, T, SIZE, MIN_ALIG, FUNC ) \
    template<class Arch> HaD \
    auto store_aligned( T *data, const Impl<T,SIZE,Arch> &impl ) -> typename std::enable_if<Arch::template Has<features::COND>::value>::type { \
        ASIMD_DEBUG_ON_OP("store_aligned",#COND) FUNC; \
    } \
    template<class P,class Arch> HaD \
    auto store( const P &data, const Impl<T,SIZE,Arch> &impl ) -> typename std::enable_if<std::is_same<T,typename std::decay<decltype(*data)>::type>::value && Arch::template Has<features::COND>::value>::type { \
        P::alignment % MIN_ALIG || P::offset % MIN_ALIG ? store_unaligned( data.get(), impl ) : store_aligned( data.get(), impl ); \
    } \
    template<class Arch> HaD \
    auto init_aligned( T *data, const Impl<T,SIZE,Arch> &impl ) -> typename std::enable_if<Arch::template Has<features::COND>::value>::type { \
        ASIMD_DEBUG_ON_OP("init_aligned",#COND) FUNC; \
    } \
    template<class P,class Arch> HaD \
    auto init( const P &data, const Impl<T,SIZE,Arch> &impl ) -> typename std::enable_if<std::is_same<T,typename std::decay<decltype(*data)>::type>::value && Arch::template Has<features::COND>::value>::type { \
        P::alignment % MIN_ALIG || P::offset % MIN_ALIG ? init_unaligned( data.get(), impl ) : init_aligned( data.get(), impl ); \
    }

#define SIMD_VEC_IMPL_REG_STORE_ALIGNED_STREAM( COND, T, SIZE, MIN_ALIG, FUNC ) \
    template<class Arch> HaD \
    auto store_aligned_stream( T *data, const Impl<T,SIZE,Arch> &impl ) -> typename std::enable_if<Arch::template Has<features::COND>::value>::type { \
        ASIMD_DEBUG_ON_OP("store_aligned_stream",#COND) FUNC; \
    } \
    template<class P,class Arch> HaD \
    auto store_stream( const P &data, const Impl<T,SIZE,Arch> &impl ) -> typename std::enable_if<std::is_same<T,typename std::decay<decltype(*data)>::type>::value && Arch::template Has<features::COND>::value>::type { \
        P::alignment % MIN_ALIG || P::offset % MIN_ALIG ? store_unaligned( data.get(), impl ) : store_aligned_stream( data.get(), impl ); \
    } \
    template<class Arch> HaD \
    auto init_aligned_stream( T *data, const Impl<T,SIZE,Arch> &impl ) -> typename std::enable_if<Arch::template Has<features::COND>::value>::type { \
        ASIMD_DEBUG_ON_OP("init_aligned_stream",#COND) FUNC; \
    } \
    template<class P,class Arch> HaD \
    auto init_stream( const P &data, const Impl<T,SIZE,Arch> &impl ) -> typename std::enable_if<std::is_same<T,typename std::decay<decltype(*data)>::type>::value && Arch::template Has<features::COND>::value>::type { \
        P::alignment % MIN_ALIG || P::offset % MIN_ALIG ? init_unaligned( data.get(), impl ) : init_aligned_stream( data.get(), impl ); \
    }

// arithmetic operations -------------------------------------------------------------
#define SIMD_VEC_IMPL_ARITHMETIC_OP( NAME, OP ) \
    template<class T,int size,class Arch> HaD \
    Impl<T,size,Arch> NAME( const Impl<T,size,Arch> &a, const Impl<T,size,Arch> &b ) { \
        Impl<T,size,Arch> res; \
        res.data.split.v0 = NAME( a.data.split.v0, b.data.split.v0 ); \
        res.data.split.v1 = NAME( a.data.split.v1, b.data.split.v1 ); \
        return res; \
    } \
    \
    template<class T,class Arch> HaD \
    Impl<T,1,Arch> NAME( const Impl<T,1,Arch> &a, const Impl<T,1,Arch> &b ) { \
        Impl<T,1,Arch> res; \
        res.data.values[ 0 ] = a.data.values[ 0 ] OP b.data.values[ 0 ]; \
        return res; \
    }

    SIMD_VEC_IMPL_ARITHMETIC_OP( sll, << )
    SIMD_VEC_IMPL_ARITHMETIC_OP( anb, &  )
    SIMD_VEC_IMPL_ARITHMETIC_OP( add, +  )
    SIMD_VEC_IMPL_ARITHMETIC_OP( sub, -  )
    SIMD_VEC_IMPL_ARITHMETIC_OP( mul, *  )
    SIMD_VEC_IMPL_ARITHMETIC_OP( div, /  )

#undef SIMD_VEC_IMPL_ARITHMETIC_OP

#define SIMD_VEC_IMPL_REG_ARITHMETIC_OP( COND, T, SIZE, NAME, FUNC ) \
    template<class Arch> HaD \
    auto NAME( const Impl<T,SIZE,Arch> &a, const Impl<T,SIZE,Arch> &b ) -> typename std::enable_if<Arch::template Has<features::COND>::value,Impl<T,SIZE,Arch>>::type { \
        ASIMD_DEBUG_ON_OP(#NAME,#COND) Impl<T,SIZE,Arch> res; res.data.reg = FUNC( a.data.reg, b.data.reg ); return res; \
    }

// cmp operations ------------------------------------------------------------------
#define SIMD_VEC_IMPL_CMP_OP( NAME, OP ) \
    template<class T,int size,class Arch,class I> HaD \
    Impl<I,size,Arch> NAME##_SimdVec( const Impl<T,size,Arch> &a, const Impl<T,size,Arch> &b, S<Impl<I,size,Arch>> ) { \
        Impl<I,size,Arch> res; \
        res.data.split.v0 = NAME##_SimdVec( a.data.split.v0, b.data.split.v0, S<Impl<I,a.split_size_0,Arch>>() ); \
        res.data.split.v1 = NAME##_SimdVec( a.data.split.v1, b.data.split.v1, S<Impl<I,a.split_size_1,Arch>>() ); \
        return res; \
    } \
    template<class T,class Arch,class I> HaD \
    Impl<I,1,Arch> NAME##_SimdVec( const Impl<T,1,Arch> &a, const Impl<T,1,Arch> &b, S<Impl<I,1,Arch>> ) { \
        Impl<I,1,Arch> res; \
        res.data.values[ 0 ] = a.data.values[ 0 ] OP b.data.values[ 0 ] ? ~I( 0 ) : I( 0 ); \
        return res; \
    } \
    template<class T,int size,class Arch> \
    struct Op_##NAME { \
        template<class VI> HaD \
        VI as_SimdVec( S<VI> ) const { \
            return NAME##_SimdVec( a, b, S<VI>() ); \
        } \
        \
        Impl<T,size,Arch> a, b; \
    }; \
    template<class T,int size,class Arch> HaD \
    Op_##NAME<T,size,Arch> NAME( const Impl<T,size,Arch> &a, const Impl<T,size,Arch> &b ) { \
        return { a, b }; \
    }

SIMD_VEC_IMPL_CMP_OP( lt, < )
SIMD_VEC_IMPL_CMP_OP( gt, > )

#undef SIMD_VEC_IMPL_CMP_OP

#define SIMD_VEC_IMPL_CMP_OP_SIMDVEC( COND, T, I, SIZE, NAME, FUNC ) \
    template<class Arch> HaD \
    auto NAME##_SimdVec( const Impl<T,SIZE,Arch> &a, const Impl<T,SIZE,Arch> &b, S<Impl<I,SIZE,Arch>> ) -> typename std::enable_if<Arch::template Has<features::COND>::value,Impl<I,SIZE,Arch>>::type { \
        ASIMD_DEBUG_ON_OP(#NAME,#COND) Impl<I,SIZE,Arch> res; res.data.reg = FUNC; return res; \
    }

// iota( beg ) --------------------------------------------------------------------------
template<class T,int size,class Arch> HaD
Impl<T,size,Arch> iota( T beg, S<Impl<T,size,Arch>> ) {
    Impl<T,size,Arch> res;
    res.data.split.v0 = iota( beg                                  , S<Impl<T,Impl<T,size,Arch>::split_size_0,Arch>>() );
    res.data.split.v1 = iota( beg + Impl<T,size,Arch>::split_size_0, S<Impl<T,Impl<T,size,Arch>::split_size_1,Arch>>() );
    return res;
}

template<class T,class Arch> HaD
Impl<T,1,Arch> iota( T beg, S<Impl<T,1,Arch>> ) {
    Impl<T,1,Arch> res;
    res.data.values[ 0 ] = beg;
    return res;
}

// iota( beg, mul ) ---------------------------------------------------------------------
template<class T,int size,class Arch> HaD
Impl<T,size,Arch> iota( T beg, T mul, S<Impl<T,size,Arch>> ) {
    Impl<T,size,Arch> res;
    res.data.split.v0 = iota( beg                                        , mul, S<Impl<T,Impl<T,size,Arch>::split_size_0,Arch>>() );
    res.data.split.v1 = iota( beg + Impl<T,size,Arch>::split_size_0 * mul, mul, S<Impl<T,Impl<T,size,Arch>::split_size_1,Arch>>() );
    return res;
}

template<class T,class Arch> HaD
Impl<T,1,Arch> iota( T beg, T /*mul*/, S<Impl<T,1,Arch>> ) {
    Impl<T,1,Arch> res;
    res.data.values[ 0 ] = beg;
    return res;
}

// sum -----------------------------------------------------------------------------
template<class T,int size,class Arch> HaD
T horizontal_sum( const Impl<T,size,Arch> &impl ) {
    return horizontal_sum( impl.data.split.v0 ) + horizontal_sum( impl.data.split.v1 );
}

template<class T,class Arch> HaD
T horizontal_sum( const Impl<T,1,Arch> &impl ) {
    return impl.data.values[ 0 ];
}

// scatter/gather -----------------------------------------------------------------------
template<class G,class V,class T,int size,class Arch> HaD
void scatter( G *ptr, const V &ind, const Impl<T,size,Arch> &vec ) {
    scatter( ptr, ind.data.split.v0, vec.data.split.v0 );
    scatter( ptr, ind.data.split.v1, vec.data.split.v1 );
}

template<class G,class V,class T,class Arch> HaD
void scatter( G *ptr, const V &ind, const Impl<T,1,Arch> &vec ) {
    ptr[ ind.data.values[ 0 ] ] = vec.data.values[ 0 ];
}

#define SIMD_VEC_IMPL_REG_SCATTER( COND, T, I, SIZE, FUNC ) \
    template<class Arch> HaD \
    auto scatter( T *data, const Impl<I,SIZE,Arch> &ind, const Impl<T,SIZE,Arch> &vec ) -> typename std::enable_if<Arch::template Has<features::COND>::value>::type { \
        ASIMD_DEBUG_ON_OP("scatter",#COND) ; FUNC; \
    }


template<class G,class V,class T,int size,class Arch> HaD
Impl<T,size,Arch> gather( const G *data, const V &ind, S<Impl<T,size,Arch>> ) {
    Impl<T,size,Arch> res;
    res.data.split.v0 = gather( data, ind.data.split.v0, S<Impl<T,Impl<T,size,Arch>::split_size_0,Arch>>() );
    res.data.split.v1 = gather( data, ind.data.split.v1, S<Impl<T,Impl<T,size,Arch>::split_size_1,Arch>>() );
    return res;
}

template<class G,class V,class T,class Arch> HaD
Impl<T,1,Arch> gather( const G *data, const V &ind, S<Impl<T,1,Arch>> ) {
    Impl<T,1,Arch> res;
    res.data.values[ 0 ] = data[ ind.data.values[ 0 ] ];
    return res;
}

#define SIMD_VEC_IMPL_REG_GATHER( COND, T, I, SIZE, FUNC ) \
    template<class Arch> HaD \
    auto gather( const T *data, const Impl<I,SIZE,Arch> &ind, S<Impl<T,SIZE,Arch>> ) -> typename std::enable_if<Arch::template Has<features::COND>::value,Impl<T,SIZE,Arch>>::type { \
        ASIMD_DEBUG_ON_OP("gather",#COND) Impl<T,SIZE,Arch> res; res.data.reg = FUNC; return res; \
    }

// min/max ---------------------------------------------------------------------
#define SIMD_VEC_IMPL_ARITHMETIC_FUNC( NAME, HELPER ) \
    template<class T,int size,class Arch> HaD \
    Impl<T,size,Arch> NAME( const Impl<T,size,Arch> &a, const Impl<T,size,Arch> &b ) { \
        Impl<T,size,Arch> res; \
        res.data.split.v0 = NAME( a.data.split.v0, b.data.split.v0 ); \
        res.data.split.v1 = NAME( a.data.split.v1, b.data.split.v1 ); \
        return res; \
    } \
    \
    template<class T,class Arch> HaD \
    Impl<T,1,Arch> NAME( const Impl<T,1,Arch> &a, const Impl<T,1,Arch> &b ) { \
        HELPER; \
        Impl<T,1,Arch> res; \
        res.data.values[ 0 ] = NAME( a.data.values[ 0 ], b.data.values[ 0 ] ); \
        return res; \
    }

    SIMD_VEC_IMPL_ARITHMETIC_FUNC( min, using std::min )
    SIMD_VEC_IMPL_ARITHMETIC_FUNC( max, using std::max )
#undef SIMD_VEC_IMPL_ARITHMETIC_FUNC


} // namespace SimdVecImpl
} // namespace asimd

