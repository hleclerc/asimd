#pragma once

#include "SimdMaskImpl_Generic.h"
#include "ASIMD_DEBUG_ON_OP.h"
#include "../support/S.h"

namespace asimd {
template<class T,int size,class Arch> struct SimdVec;

namespace internal {

// SimdVecImpl ---------------------------------------------------------
/// Splittable version
template<class T,int size,class Arch>
struct SimdVecImpl {
    static constexpr int split_size_0 = prev_pow_2( size );
    static constexpr int split_size_1 = size - split_size_0;
    struct Split {
        SimdVecImpl<T,split_size_0,Arch> v0;
        SimdVecImpl<T,split_size_1,Arch> v1;
    };
    union {
        T     values[ size ];
        Split split;
    } data;
};

/// Atomic version
template<class T,class Arch>
struct SimdVecImpl<T,1,Arch> {
    union {
        T values[ 1 ];
    } data;
};

/// Helper to make Impl with a register
#define SIMD_VEC_IMPL_REG( COND, T, SIZE, TREG ) \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) \
    struct SimdVecImpl<T,SIZE,Arch> { \
        static constexpr int split_size_0 = prev_pow_2( SIZE ); \
        static constexpr int split_size_1 = SIZE - split_size_0; \
        struct Split { \
            SimdVecImpl<T,split_size_0,Arch> v0; \
            SimdVecImpl<T,split_size_1,Arch> v1; \
        }; \
        union { \
            T     values[ SIZE ]; \
            Split split; \
            TREG  reg; \
        } data; \
    }

// at ------------------------------------------------------------------------
template<class T,int size,class Arch> HaD
const T &at( const SimdVecImpl<T,size,Arch> &vec, int i ) {
    return vec.data.values[ i ];
}

template<class T,int size,class Arch> HaD
T &at( SimdVecImpl<T,size,Arch> &vec, int i ) {
    return vec.data.values[ i ];
}

// init ----------------------------------------------------------------------
template<class T,int size,class Arch,class G> HaD
void init_sc( SimdVecImpl<T,size,Arch> &vec, G a, G b, G c, G d, G e, G f, G g, G h ) {
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
void init_sc( SimdVecImpl<T,size,Arch> &vec, G a, G b, G c, G d, G e ) {
    vec.data.values[ 0 ] = a;
    vec.data.values[ 1 ] = b;
    vec.data.values[ 2 ] = c;
    vec.data.values[ 3 ] = d;
    vec.data.values[ 4 ] = e;
}

template<class T,int size,class Arch,class G> HaD
void init_sc( SimdVecImpl<T,size,Arch> &vec, G a, G b, G c, G d ) {
    vec.data.values[ 0 ] = a;
    vec.data.values[ 1 ] = b;
    vec.data.values[ 2 ] = c;
    vec.data.values[ 3 ] = d;
}

template<class T,int size,class Arch,class G> HaD
void init_sc( SimdVecImpl<T,size,Arch> &vec, G a, G b ) {
    vec.data.values[ 0 ] = a;
    vec.data.values[ 1 ] = b;
}

template<class T,int size,class Arch,class G> HaD
void init_sc( SimdVecImpl<T,size,Arch> &vec, G a ) {
    for( int i = 0; i < size; ++i )
        vec.data.values[ i ] = a;
}

template<class T,int size,int part,class Arch> HaD
void init_sc( SimdVecImpl<T,size,Arch> &vec, SimdVecImpl<T,part,Arch> a, SimdVecImpl<T,size-part,Arch> b ) {
    vec.data.split.v0 = a;
    vec.data.split.v1 = b;
}

#define SIMD_VEC_IMPL_REG_INIT_1( COND, T, SIZE, FUNC ) \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) HaD \
    void init_sc( SimdVecImpl<T,SIZE,Arch> &vec, T a ) { \
        ASIMD_DEBUG_ON_OP("init_sc_1",#COND,#FUNC) vec.data.reg = FUNC; \
    }

#define SIMD_VEC_IMPL_REG_INIT_2( COND, T, SIZE, FUNC ) \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) HaD \
    void init_sc( SimdVecImpl<T,SIZE,Arch> &vec, T a, T b ) { \
        ASIMD_DEBUG_ON_OP("init_sc_2",#COND,#FUNC) vec.data.reg = FUNC; \
    }

#define SIMD_VEC_IMPL_REG_INIT_4( COND, T, SIZE, FUNC ) \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) HaD \
    void init_sc( SimdVecImpl<T,SIZE,Arch> &vec, T a, T b, T c, T d ) { \
        ASIMD_DEBUG_ON_OP("init_sc_4",#COND,#FUNC) vec.data.reg = FUNC; \
    }

#define SIMD_VEC_IMPL_REG_INIT_8( COND, T, SIZE, FUNC ) \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) HaD \
    void init_sc( SimdVecImpl<T,SIZE,Arch> &vec, T a, T b, T c, T d, T e, T f, T g, T h ) { \
        ASIMD_DEBUG_ON_OP("init_sc_8",#COND,#FUNC) vec.data.reg = FUNC; \
    }

#define SIMD_VEC_IMPL_REG_INIT_16( COND, T, SIZE, FUNC ) \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) HaD \
    void init_sc( SimdVecImpl<T,SIZE,Arch> &vec, T a, T b, T c, T d, T e, T f, T g, T h, T i, T j, T k, T l, T m, T n, T o, T p ) { \
        ASIMD_DEBUG_ON_OP("init_sc_16",#COND,#FUNC) vec.data.reg = FUNC; \
    }

// prefetch ----------------------------------------------------------------------
template<int len,class N_len,class S_Arch>
void prefetch( const void *, N_len, S_Arch ) {
}

// load_unaligned ----------------------------------------------------------------------------
template<class G,class T,int size,class Arch> HaD
SimdVecImpl<T,size,Arch> load_unaligned( const G *data, S<SimdVecImpl<T,size,Arch>> ) {
    SimdVecImpl<T,size,Arch> res;
    res.data.split.v0 = load_unaligned( data                                         , S<SimdVecImpl<T,SimdVecImpl<T,size,Arch>::split_size_0,Arch>>() );
    res.data.split.v1 = load_unaligned( data + SimdVecImpl<T,size,Arch>::split_size_0, S<SimdVecImpl<T,SimdVecImpl<T,size,Arch>::split_size_1,Arch>>() );
    return res;
}

template<class G,class T,class Arch> HaD
SimdVecImpl<T,1,Arch> load_unaligned( const G *data, S<SimdVecImpl<T,1,Arch>> ) {
    SimdVecImpl<T,1,Arch> res;
    res.data.values[ 0 ] = *data;
    return res;
}

#define SIMD_VEC_IMPL_REG_LOAD_UNALIGNED( COND, T, SIZE, FUNC ) \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) HaD \
    SimdVecImpl<T,SIZE,Arch> load_unaligned( const T *data, S<SimdVecImpl<T,SIZE,Arch>> ) { \
        ASIMD_DEBUG_ON_OP("load_unaligned",#COND,#FUNC) SimdVecImpl<T,SIZE,Arch> res; res.data.reg = FUNC; return res; \
    }

#define SIMD_VEC_IMPL_REG_LOAD_FOT( COND, T, G, SIZE, FUNC ) \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) HaD \
    SimdVecImpl<T,SIZE,Arch> load_unaligned( const G *data, S<SimdVecImpl<T,SIZE,Arch>> ) { \
        ASIMD_DEBUG_ON_OP("store_unaligned_fot",#COND,#FUNC) SimdVecImpl<T,SIZE,Arch> res; res.data.reg = FUNC; return res; \
    }

// load_aligned ---------------------------------------------------------------------------------
template<class G,class T,int size,class Arch> HaD
SimdVecImpl<T,size,Arch> load_aligned( const G *data, S<SimdVecImpl<T,size,Arch>> ) {
    SimdVecImpl<T,size,Arch> res;
    res.data.split.v0 = load_aligned( data                                         , S<SimdVecImpl<T,SimdVecImpl<T,size,Arch>::split_size_0,Arch>>() );
    res.data.split.v1 = load_aligned( data + SimdVecImpl<T,size,Arch>::split_size_0, S<SimdVecImpl<T,SimdVecImpl<T,size,Arch>::split_size_1,Arch>>() );
    return res;
}

template<class G,class T,class Arch> HaD
SimdVecImpl<T,1,Arch> load_aligned( const G *data, S<SimdVecImpl<T,1,Arch>> ) {
    SimdVecImpl<T,1,Arch> res;
    res.data.values[ 0 ] = *data;
    return res;
}

template<class G,class T,int size,class Arch> HaD
SimdVecImpl<T,size,Arch> load_aligned_stream( const G *data, S<SimdVecImpl<T,size,Arch>> ) {
    SimdVecImpl<T,size,Arch> res;
    res.data.split.v0 = load_aligned_stream( data                                         , S<SimdVecImpl<T,SimdVecImpl<T,size,Arch>::split_size_0,Arch>>() );
    res.data.split.v1 = load_aligned_stream( data + SimdVecImpl<T,size,Arch>::split_size_0, S<SimdVecImpl<T,SimdVecImpl<T,size,Arch>::split_size_1,Arch>>() );
    return res;
}

template<class G,class T,class Arch> HaD
SimdVecImpl<T,1,Arch> load_aligned_stream( const G *data, S<SimdVecImpl<T,1,Arch>> ) {
    SimdVecImpl<T,1,Arch> res;
    res.data.values[ 0 ] = *data;
    return res;
}

template<class P,class T,int size,class Arch> HaD
SimdVecImpl<T,size,Arch> load( const P &data, S<SimdVecImpl<T,size,Arch>> ) {
    SimdVecImpl<T,size,Arch> res;
    res.data.split.v0 = load( data                                              , S<SimdVecImpl<T,SimdVecImpl<T,size,Arch>::split_size_0,Arch>>() );
    res.data.split.v1 = load( data + N<SimdVecImpl<T,size,Arch>::split_size_0>(), S<SimdVecImpl<T,SimdVecImpl<T,size,Arch>::split_size_1,Arch>>() );
    return res;
}

template<class P,class T,class Arch> HaD
SimdVecImpl<T,1,Arch> load( const P &data, S<SimdVecImpl<T,1,Arch>> ) {
    SimdVecImpl<T,1,Arch> res;
    res.data.values[ 0 ] = *data.get();
    return res;
}

template<class P,class T,int size,class Arch> HaD
SimdVecImpl<T,size,Arch> load_stream( const P &data, S<SimdVecImpl<T,size,Arch>> ) {
    SimdVecImpl<T,size,Arch> res;
    res.data.split.v0 = load_stream( data                                              , S<SimdVecImpl<T,SimdVecImpl<T,size,Arch>::split_size_0,Arch>>() );
    res.data.split.v1 = load_stream( data + N<SimdVecImpl<T,size,Arch>::split_size_0>(), S<SimdVecImpl<T,SimdVecImpl<T,size,Arch>::split_size_1,Arch>>() );
    return res;
}

template<class P,class T,class Arch> HaD
SimdVecImpl<T,1,Arch> load_stream( const P &data, S<SimdVecImpl<T,1,Arch>> ) {
    SimdVecImpl<T,1,Arch> res;
    res.data.values[ 0 ] = *data.get();
    return res;
}

#define SIMD_VEC_IMPL_REG_LOAD_ALIGNED( COND, T, SIZE, MIN_ALIG, FUNC ) \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) HaD \
    SimdVecImpl<T,SIZE,Arch> load_aligned( const T *data, S<SimdVecImpl<T,SIZE,Arch>> ) { \
        ASIMD_DEBUG_ON_OP("load_aligned",#COND,#FUNC) SimdVecImpl<T,SIZE,Arch> res; res.data.reg = FUNC; return res; \
    } \
    template<class P,class Arch> HaD \
    SimdVecImpl<T,SIZE,Arch> load( const P &data, S<SimdVecImpl<T,SIZE,Arch>> s ) requires ( std::is_same<T,typename std::decay<decltype(*data)>::type>::value && Arch::template Has<features::COND>::value ) { \
        return P::alignment % MIN_ALIG || P::offset % MIN_ALIG ? load_unaligned( data.get(), s ) : load_aligned( data.get(), s ); \
    }

#define SIMD_VEC_IMPL_REG_LOAD_ALIGNED_STREAM( COND, T, SIZE, MIN_ALIG, FUNC ) \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) HaD \
    SimdVecImpl<T,SIZE,Arch> load_aligned_stream( const T *data, S<SimdVecImpl<T,SIZE,Arch>> ) { \
        ASIMD_DEBUG_ON_OP("load_aligned_stream",#COND,#FUNC) SimdVecImpl<T,SIZE,Arch> res; res.data.reg = FUNC; return res; \
    } \
    template<class P,class Arch> HaD \
    SimdVecImpl<T,SIZE,Arch> load_stream( const P &data, S<SimdVecImpl<T,SIZE,Arch>> s ) requires ( std::is_same<T,typename std::decay<decltype(*data)>::type>::value && Arch::template Has<features::COND>::value ) { \
        return P::alignment % MIN_ALIG || P::offset % MIN_ALIG ? load_unaligned( data.get(), s ) : load_aligned_stream( data.get(), s ); \
    }

#define SIMD_VEC_IMPL_REG_LOAD_ALIGNED_FOT( COND, T, G, SIZE, MIN_ALIG, FUNC ) \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) HaD \
    SimdVecImpl<T,SIZE,Arch> load_aligned( const G *data, S<SimdVecImpl<T,SIZE,Arch>> ) { \
        ASIMD_DEBUG_ON_OP("load_aligned_fot",#COND,#FUNC) SimdVecImpl<T,SIZE,Arch> res; res.data.reg = FUNC; return res; \
    } \
    template<class P,int a> requires ( std::is_same<G,typename std::decay<decltype(*data)>::type>::value && Arch::template Has<features::COND>::value ) HaD \
    SimdVecImpl<T,SIZE,Arch> load( const P &data, S<SimdVecImpl<T,SIZE,Arch>> s ) { \
        return P::alignment % MIN_ALIG || P::offset % MIN_ALIG ? load_unaligned( data.get(), s ) : load_aligned( data.get(), s ); \
    }

// store and init unaligned -----------------------------------------------------------------------
template<class G,class T,int size,class Arch> HaD
void store_unaligned( G *data, const SimdVecImpl<T,size,Arch> &impl ) {
    store_unaligned( data                    , impl.data.split.v0 );
    store_unaligned( data + impl.split_size_0, impl.data.split.v1 );
}

template<class G,class T,class Arch> HaD
void store_unaligned( G *data, const SimdVecImpl<T,1,Arch> &impl ) {
    *data = impl.data.values[ 0 ];
}

template<class G,class T,int size,class Arch> HaD
void init_unaligned( G *data, const SimdVecImpl<T,size,Arch> &impl ) {
    init_unaligned( data                    , impl.data.split.v0 );
    init_unaligned( data + impl.split_size_0, impl.data.split.v1 );
}

template<class G,class T,class Arch> HaD
void init_unaligned( G *data, const SimdVecImpl<T,1,Arch> &impl ) {
    new ( data ) G( impl.data.values[ 0 ] );
}

#define SIMD_VEC_IMPL_REG_STORE_UNALIGNED( COND, T, SIZE, FUNC ) \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) HaD \
    void store_unaligned( T *data, const SimdVecImpl<T,SIZE,Arch> &impl ) { \
        ASIMD_DEBUG_ON_OP("store_unaligned",#COND,#FUNC) FUNC; \
    } \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) HaD \
    void init_unaligned( T *data, const SimdVecImpl<T,SIZE,Arch> &impl ) { \
        ASIMD_DEBUG_ON_OP("init_unaligned",#COND,#FUNC) FUNC; \
    }

// store and init aligned -----------------------------------------------------------------------
template<class G,class T,int size,class Arch> HaD
void store_aligned( G *data, const SimdVecImpl<T,size,Arch> &impl ) {
    store_aligned( data                    , impl.data.split.v0 );
    store_aligned( data + impl.split_size_0, impl.data.split.v1 );
}

template<class G,class T,class Arch> HaD
void store_aligned( G *data, const SimdVecImpl<T,1,Arch> &impl ) {
    *data = impl.data.values[ 0 ];
}

template<class P,class T,int size,class Arch> HaD
void store( const P &data, const SimdVecImpl<T,size,Arch> &impl ) {
    store( data                         , impl.data.split.v0 );
    store( data + N<impl.split_size_0>(), impl.data.split.v1 );
}

template<class P,class T,class Arch> HaD
void store( const P &data, const SimdVecImpl<T,1,Arch> &impl ) {
    *data = impl.data.values[ 0 ];
}

template<class G,class T,int size,class Arch> HaD
void store_aligned_stream( G *data, const SimdVecImpl<T,size,Arch> &impl ) {
    store_aligned_stream( data                    , impl.data.split.v0 );
    store_aligned_stream( data + impl.split_size_0, impl.data.split.v1 );
}

template<class G,class T,class Arch> HaD
void store_aligned_stream( G *data, const SimdVecImpl<T,1,Arch> &impl ) {
    *data = impl.data.values[ 0 ];
}

template<class P,class T,int size,class Arch> HaD
void store_stream( const P &data, const SimdVecImpl<T,size,Arch> &impl ) {
    store_stream( data                                              , impl.data.split.v0 );
    store_stream( data + N<SimdVecImpl<T,size,Arch>::split_size_0>(), impl.data.split.v1 );
}

template<class P,class T,class Arch> HaD
void store_stream( const P &data, const SimdVecImpl<T,1,Arch> &impl ) {
    *data = impl.data.values[ 0 ];
}

template<class G,class T,int size,class Arch> HaD
void init_aligned( G *data, const SimdVecImpl<T,size,Arch> &impl ) {
    init_aligned( data                    , impl.data.split.v0 );
    init_aligned( data + impl.split_size_0, impl.data.split.v1 );
}

template<class G,class T,class Arch> HaD
void init_aligned( G *data, const SimdVecImpl<T,1,Arch> &impl ) {
    new ( data ) G( impl.data.values[ 0 ] );
}

template<class P,class T,int size,class Arch> HaD
void init( const P &data, const SimdVecImpl<T,size,Arch> &impl ) {
    init( data                         , impl.data.split.v0 );
    init( data + N<impl.split_size_0>(), impl.data.split.v1 );
}

template<class P,class T,class Arch> HaD
void init( const P &data, const SimdVecImpl<T,1,Arch> &impl ) {
    using G = typename std::decay<decltype(*data)>::type;
    new ( data.get() ) G( impl.data.values[ 0 ] );
}

template<class G,class T,int size,class Arch> HaD
void init_aligned_stream( G *data, const SimdVecImpl<T,size,Arch> &impl ) {
    init_aligned_stream( data                    , impl.data.split.v0 );
    init_aligned_stream( data + impl.split_size_0, impl.data.split.v1 );
}

template<class G,class T,class Arch> HaD
void init_aligned_stream( G *data, const SimdVecImpl<T,1,Arch> &impl ) {
    new ( data ) G( impl.data.values[ 0 ] );
}

template<class P,class T,int size,class Arch> HaD
void init_stream( const P &data, const SimdVecImpl<T,size,Arch> &impl ) {
    init_stream( data                         , impl.data.split.v0 );
    init_stream( data + N<impl.split_size_0>(), impl.data.split.v1 );
}

template<class P,class T,class Arch> HaD
void init_stream( const P &data, const SimdVecImpl<T,1,Arch> &impl ) {
    using G = typename std::decay<decltype(*data)>::type;
    new ( data.get() ) G( impl.data.values[ 0 ] );
}

#define SIMD_VEC_IMPL_REG_STORE_ALIGNED( COND, T, SIZE, MIN_ALIG, FUNC ) \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) HaD \
    void store_aligned( T *data, const SimdVecImpl<T,SIZE,Arch> &impl ) { \
        ASIMD_DEBUG_ON_OP("store_aligned",#COND,#FUNC) FUNC; \
    } \
    template<class P,class Arch> HaD \
    void store( const P &data, const SimdVecImpl<T,SIZE,Arch> &impl ) requires ( std::is_same<T,typename std::decay<decltype(*data)>::type>::value && Arch::template Has<features::COND>::value ) { \
        P::alignment % MIN_ALIG || P::offset % MIN_ALIG ? store_unaligned( data.get(), impl ) : store_aligned( data.get(), impl ); \
    } \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) HaD \
    void init_aligned( T *data, const SimdVecImpl<T,SIZE,Arch> &impl ) { \
        ASIMD_DEBUG_ON_OP("init_aligned",#COND,#FUNC) FUNC; \
    } \
    template<class P,class Arch> HaD \
    void init( const P &data, const SimdVecImpl<T,SIZE,Arch> &impl ) requires ( std::is_same<T,typename std::decay<decltype(*data)>::type>::value && Arch::template Has<features::COND>::value ) { \
        P::alignment % MIN_ALIG || P::offset % MIN_ALIG ? init_unaligned( data.get(), impl ) : init_aligned( data.get(), impl ); \
    }

#define SIMD_VEC_IMPL_REG_STORE_ALIGNED_STREAM( COND, T, SIZE, MIN_ALIG, FUNC ) \
    template<class Arch> requires( Arch::template Has<features::COND>::value ) HaD \
    void store_aligned_stream( T *data, const SimdVecImpl<T,SIZE,Arch> &impl ) { \
        ASIMD_DEBUG_ON_OP("store_aligned_stream",#COND,#FUNC) FUNC; \
    } \
    template<class P,class Arch> HaD \
    void store_stream( const P &data, const SimdVecImpl<T,SIZE,Arch> &impl ) requires ( std::is_same<T,typename std::decay<decltype(*data)>::type>::value && Arch::template Has<features::COND>::value ) { \
        P::alignment % MIN_ALIG || P::offset % MIN_ALIG ? store_unaligned( data.get(), impl ) : store_aligned_stream( data.get(), impl ); \
    } \
    template<class Arch> requires( Arch::template Has<features::COND>::value ) HaD \
    void init_aligned_stream( T *data, const SimdVecImpl<T,SIZE,Arch> &impl ) { \
        ASIMD_DEBUG_ON_OP("init_aligned_stream",#COND,#FUNC) FUNC; \
    } \
    template<class P,class Arch> HaD \
    void init_stream( const P &data, const SimdVecImpl<T,SIZE,Arch> &impl ) requires( std::is_same<T,typename std::decay<decltype(*data)>::type>::value && Arch::template Has<features::COND>::value ) { \
        P::alignment % MIN_ALIG || P::offset % MIN_ALIG ? init_unaligned( data.get(), impl ) : init_aligned_stream( data.get(), impl ); \
    }

// arithmetic operations -------------------------------------------------------------
#define SIMD_VEC_IMPL_ARITHMETIC_OP( NAME, OP ) \
    template<class T,int size,class Arch> HaD \
    SimdVecImpl<T,size,Arch> NAME( const SimdVecImpl<T,size,Arch> &a, const SimdVecImpl<T,size,Arch> &b ) { \
        SimdVecImpl<T,size,Arch> res; \
        res.data.split.v0 = NAME( a.data.split.v0, b.data.split.v0 ); \
        res.data.split.v1 = NAME( a.data.split.v1, b.data.split.v1 ); \
        return res; \
    } \
    \
    template<class T,class Arch> HaD \
    SimdVecImpl<T,1,Arch> NAME( const SimdVecImpl<T,1,Arch> &a, const SimdVecImpl<T,1,Arch> &b ) { \
        SimdVecImpl<T,1,Arch> res; \
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
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) HaD \
    SimdVecImpl<T,SIZE,Arch> NAME( const SimdVecImpl<T,SIZE,Arch> &a, const SimdVecImpl<T,SIZE,Arch> &b ) { \
        ASIMD_DEBUG_ON_OP(#NAME,#COND,#FUNC) SimdVecImpl<T,SIZE,Arch> res; res.data.reg = FUNC( a.data.reg, b.data.reg ); return res; \
    }

// cmp operations ------------------------------------------------------------------
#define SIMD_VEC_IMPL_CMP_OP( NAME, OP ) \
    /* _as_a_simd_mask */ \
    template<class T,int size,class Arch> HaD \
    SimdMaskImpl<size,1,Arch> NAME##_as_a_simd_mask( const SimdVecImpl<T,size,Arch> &a, const SimdVecImpl<T,size,Arch> &b ) { \
        SimdMaskImpl<size,1,Arch> res; \
        res.data.split.v0 = NAME##_as_a_simd_mask( a.data.split.v0, b.data.split.v0 ); \
        res.data.split.v1 = NAME##_as_a_simd_mask( a.data.split.v1, b.data.split.v1 ); \
        return res; \
    } \
    template<class T,class Arch> HaD \
    SimdMaskImpl<8,1,Arch> NAME##_as_a_simd_mask( const SimdVecImpl<T,8,Arch> &a, const SimdVecImpl<T,8,Arch> &b ) { \
        SimdMaskImpl<8,1,Arch> res; \
        res.data.values.set_values( \
            a.data.values[ 0 ] OP b.data.values[ 0 ], a.data.values[ 1 ] OP b.data.values[ 1 ], a.data.values[ 2 ] OP b.data.values[ 2 ], a.data.values[ 3 ] OP b.data.values[ 3 ], \
            a.data.values[ 4 ] OP b.data.values[ 4 ], a.data.values[ 5 ] OP b.data.values[ 5 ], a.data.values[ 6 ] OP b.data.values[ 6 ], a.data.values[ 7 ] OP b.data.values[ 7 ]  \
        ); \
        return res; \
    } \
    template<class T,class Arch> HaD \
    SimdMaskImpl<4,1,Arch> NAME##_as_a_simd_mask( const SimdVecImpl<T,4,Arch> &a, const SimdVecImpl<T,4,Arch> &b ) { \
        SimdMaskImpl<4,1,Arch> res; \
        res.data.values.set_values( \
            a.data.values[ 0 ] OP b.data.values[ 0 ], a.data.values[ 1 ] OP b.data.values[ 1 ], a.data.values[ 2 ] OP b.data.values[ 2 ], a.data.values[ 3 ] OP b.data.values[ 3 ] \
        ); \
        return res; \
    } \
    template<class T,class Arch> HaD \
    SimdMaskImpl<2,1,Arch> NAME##_as_a_simd_mask( const SimdVecImpl<T,2,Arch> &a, const SimdVecImpl<T,2,Arch> &b ) { \
        SimdMaskImpl<2,1,Arch> res; \
        res.data.values.set_values( \
            a.data.values[ 0 ] OP b.data.values[ 0 ], a.data.values[ 1 ] OP b.data.values[ 1 ] \
        ); \
        return res; \
    } \
    template<class T,class Arch> HaD \
    SimdMaskImpl<1,1,Arch> NAME##_as_a_simd_mask( const SimdVecImpl<T,1,Arch> &a, const SimdVecImpl<T,1,Arch> &b ) { \
        SimdMaskImpl<2,1,Arch> res; \
        res.data.values.set_values( \
            a.data.values[ 0 ] OP b.data.values[ 0 ] \
        ); \
        return res; \
    } \
    /* _as_a_simd_vec */ \
    template<class T,int size,class Arch,class I> HaD \
    SimdVecImpl<I,size,Arch> NAME##_as_a_simd_vec( const SimdVecImpl<T,size,Arch> &a, const SimdVecImpl<T,size,Arch> &b, S<SimdVecImpl<I,size,Arch>> ) { \
        SimdVecImpl<I,size,Arch> res; \
        res.data.split.v0 = NAME##_as_a_simd_vec( a.data.split.v0, b.data.split.v0, S<SimdVecImpl<I,a.split_size_0,Arch>>() ); \
        res.data.split.v1 = NAME##_as_a_simd_vec( a.data.split.v1, b.data.split.v1, S<SimdVecImpl<I,a.split_size_1,Arch>>() ); \
        return res; \
    } \
    template<class T,class Arch,class I> HaD \
    SimdVecImpl<I,1,Arch> NAME##_as_a_simd_vec( const SimdVecImpl<T,1,Arch> &a, const SimdVecImpl<T,1,Arch> &b, S<SimdVecImpl<I,1,Arch>> ) { \
        SimdVecImpl<I,1,Arch> res; \
        res.data.values[ 0 ] = a.data.values[ 0 ] OP b.data.values[ 0 ] ? ~I( 0 ) : I( 0 ); \
        return res; \
    } \
    template<class T,int s,class Arch> \
    struct Op_##NAME { \
        bool operator[]( PI index ) const { return a.data.values[ index ] OP b.data.values[ index ]; } \
        PI size() const { return s; } \
        \
        SimdVecImpl<T,s,Arch> a, b; \
    }; \
    \
    template<class T,int size,class Arch> HaD \
    Op_##NAME<T,size,Arch> NAME( const SimdVecImpl<T,size,Arch> &a, const SimdVecImpl<T,size,Arch> &b ) { \
        return { a, b }; \
    }

SIMD_VEC_IMPL_CMP_OP( lt, < )
SIMD_VEC_IMPL_CMP_OP( gt, > )

#undef SIMD_VEC_IMPL_CMP_OP

#define SIMD_VEC_IMPL_CMP_OP_SIMDVEC( COND, T, NB_ITEMS, ITEM_SIZE, NAME, FUNC ) \
    template<class Arch> requires( Arch::template Has<features::COND>::value ) HaD \
    auto NAME##_as_a_simd_mask( const SimdVecImpl<T,NB_ITEMS,Arch> &a, const SimdVecImpl<T,NB_ITEMS,Arch> &b ) { \
        ASIMD_DEBUG_ON_OP(#NAME,#COND,#FUNC) SimdMaskImpl<NB_ITEMS,ITEM_SIZE,Arch> res; res.data.reg = FUNC; return res; \
    }

#define SIMD_VEC_IMPL_CMP_OP_SIMDVEC_VEC( COND, T, I, SIZE, NAME, FUNC ) \
    template<class Arch> requires( Arch::template Has<features::COND>::value ) HaD \
    auto NAME##_as_a_simd_vec( const SimdVecImpl<T,SIZE,Arch> &a, const SimdVecImpl<T,SIZE,Arch> &b, S<SimdVecImpl<I,SIZE,Arch>> ) { \
        ASIMD_DEBUG_ON_OP(#NAME,#COND,#FUNC) SimdVecImpl<I,SIZE,Arch> res; res.data.reg = FUNC; return res; \
    }

// iota( beg ) --------------------------------------------------------------------------
template<class T,int size,class Arch> HaD
SimdVecImpl<T,size,Arch> iota( T beg, S<SimdVecImpl<T,size,Arch>> ) {
    SimdVecImpl<T,size,Arch> res;
    res.data.split.v0 = iota( beg                                         , S<SimdVecImpl<T,SimdVecImpl<T,size,Arch>::split_size_0,Arch>>() );
    res.data.split.v1 = iota( beg + SimdVecImpl<T,size,Arch>::split_size_0, S<SimdVecImpl<T,SimdVecImpl<T,size,Arch>::split_size_1,Arch>>() );
    return res;
}

template<class T,class Arch> HaD
SimdVecImpl<T,1,Arch> iota( T beg, S<SimdVecImpl<T,1,Arch>> ) {
    SimdVecImpl<T,1,Arch> res;
    res.data.values[ 0 ] = beg;
    return res;
}

// iota( beg, mul ) ---------------------------------------------------------------------
template<class T,int size,class Arch> HaD
SimdVecImpl<T,size,Arch> iota( T beg, T mul, S<SimdVecImpl<T,size,Arch>> ) {
    SimdVecImpl<T,size,Arch> res;
    res.data.split.v0 = iota( beg                                               , mul, S<SimdVecImpl<T,SimdVecImpl<T,size,Arch>::split_size_0,Arch>>() );
    res.data.split.v1 = iota( beg + SimdVecImpl<T,size,Arch>::split_size_0 * mul, mul, S<SimdVecImpl<T,SimdVecImpl<T,size,Arch>::split_size_1,Arch>>() );
    return res;
}

template<class T,class Arch> HaD
SimdVecImpl<T,1,Arch> iota( T beg, T /*mul*/, S<SimdVecImpl<T,1,Arch>> ) {
    SimdVecImpl<T,1,Arch> res;
    res.data.values[ 0 ] = beg;
    return res;
}

// sum -----------------------------------------------------------------------------
template<class T,int size,class Arch> HaD
T horizontal_sum( const SimdVecImpl<T,size,Arch> &impl ) {
    return horizontal_sum( impl.data.split.v0 ) + horizontal_sum( impl.data.split.v1 );
}

template<class T,class Arch> HaD
T horizontal_sum( const SimdVecImpl<T,1,Arch> &impl ) {
    return impl.data.values[ 0 ];
}

// scatter/gather -----------------------------------------------------------------------
template<class G,class V,class T,int size,class Arch> HaD
void scatter( G *ptr, const V &ind, const SimdVecImpl<T,size,Arch> &vec ) {
    scatter( ptr, ind.data.split.v0, vec.data.split.v0 );
    scatter( ptr, ind.data.split.v1, vec.data.split.v1 );
}

template<class G,class V,class T,class Arch> HaD
void scatter( G *ptr, const V &ind, const SimdVecImpl<T,1,Arch> &vec ) {
    ptr[ ind.data.values[ 0 ] ] = vec.data.values[ 0 ];
}

#define SIMD_VEC_IMPL_REG_SCATTER( COND, T, I, SIZE, FUNC ) \
    template<class Arch> requires( Arch::template Has<features::COND>::value ) HaD \
    auto scatter( T *data, const SimdVecImpl<I,SIZE,Arch> &ind, const SimdVecImpl<T,SIZE,Arch> &vec ) { \
        ASIMD_DEBUG_ON_OP("scatter",#COND,#FUNC); FUNC; \
    }


template<class G,class V,class T,int size,class Arch> HaD
SimdVecImpl<T,size,Arch> gather( const G *data, const V &ind, S<SimdVecImpl<T,size,Arch>> ) {
    SimdVecImpl<T,size,Arch> res;
    res.data.split.v0 = gather( data, ind.data.split.v0, S<SimdVecImpl<T,SimdVecImpl<T,size,Arch>::split_size_0,Arch>>() );
    res.data.split.v1 = gather( data, ind.data.split.v1, S<SimdVecImpl<T,SimdVecImpl<T,size,Arch>::split_size_1,Arch>>() );
    return res;
}

template<class G,class V,class T,class Arch> HaD
SimdVecImpl<T,1,Arch> gather( const G *data, const V &ind, S<SimdVecImpl<T,1,Arch>> ) {
    SimdVecImpl<T,1,Arch> res;
    res.data.values[ 0 ] = data[ ind.data.values[ 0 ] ];
    return res;
}

#define SIMD_VEC_IMPL_REG_GATHER( COND, T, I, SIZE, FUNC ) \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) HaD \
    SimdVecImpl<T,SIZE,Arch> gather( const T *data, const SimdVecImpl<I,SIZE,Arch> &ind, S<SimdVecImpl<T,SIZE,Arch>> ) { \
        ASIMD_DEBUG_ON_OP("gather",#COND,#FUNC) SimdVecImpl<T,SIZE,Arch> res; res.data.reg = FUNC; return res; \
    }

// min/max ---------------------------------------------------------------------
#define SIMD_VEC_IMPL_ARITHMETIC_FUNC( NAME, HELPER ) \
    template<class T,int size,class Arch> HaD \
    SimdVecImpl<T,size,Arch> NAME( const SimdVecImpl<T,size,Arch> &a, const SimdVecImpl<T,size,Arch> &b ) { \
        SimdVecImpl<T,size,Arch> res; \
        res.data.split.v0 = NAME( a.data.split.v0, b.data.split.v0 ); \
        res.data.split.v1 = NAME( a.data.split.v1, b.data.split.v1 ); \
        return res; \
    } \
    \
    template<class T,class Arch> HaD \
    SimdVecImpl<T,1,Arch> NAME( const SimdVecImpl<T,1,Arch> &a, const SimdVecImpl<T,1,Arch> &b ) { \
        HELPER; \
        SimdVecImpl<T,1,Arch> res; \
        res.data.values[ 0 ] = NAME( a.data.values[ 0 ], b.data.values[ 0 ] ); \
        return res; \
    }

    SIMD_VEC_IMPL_ARITHMETIC_FUNC( min, using std::min )
    SIMD_VEC_IMPL_ARITHMETIC_FUNC( max, using std::max )
#undef SIMD_VEC_IMPL_ARITHMETIC_FUNC


} // namespace internal
} // namespace asimd

