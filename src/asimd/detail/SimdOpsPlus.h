#pragma once

// =====================================================================================
// THE PRIMITIVES BEYOND LOAD/STORE/ARITHMETIC.
//
// `SimdVec` knows how to load, store, add, compare and gather. The five operations here are what
// branchless vector code needs on top of that:
//
//   `fma`             fused multiply-add -- a dot product is a chain of these
//   `to_bits`         the sign mask AS AN INTEGER. It carries the `ctz`, the `popcount` and the
//                     rotations, hence everything that replaces a traversal
//   `mask_from_bits`  the dual: one `kmovb` where `eq( iota, i )` costs two instructions
//   `select`          blend two vectors according to a mask
//   `permute`         VARIABLE-INDEX permutation, by far the costliest one to get right
//
// plus `eq` / `ge` (only `lt` and `gt` existed), `bcast_lane` (broadcast a compile-time lane) and
// the `-` `*` `/` operators -- the underlying operations were already there, only the facades
// were missing.
//
// EVERY OPERATION GOES THROUGH `Selection.h`. A generic form is registered at rank `GENERIC` and
// is always available, so there is always an answer; `SimdOpsPlus_X86.h` registers register-level
// forms at `REGISTER` and `MASK_REGISTER`. The ranks give a TOTAL order, which is what makes an
// AVX-512 form preferable to an AVX one -- two separately written `requires` clauses cannot
// express that, they merely become ambiguous.
//
// `permute` deserves a note. Its generic form goes through MEMORY (store, index, reload), because
// a split into two half-registers cannot move a lane from one half to the other without meeting
// somewhere. That is correct everywhere and slow; the AVX2 form (`vpermps`) is a single shot. An
// ARM port will want `tbl`, an SVE one `vrgather`.
// =====================================================================================

#include "Selection.h"
#include "SimdMask.h"
#include "SimdVec.h"

#include <utility>

namespace asimd {

/// Operation tags.
namespace ops {
    struct fma {};
    struct permute {};
    struct select {};
    struct to_bits {};
    struct mask_from_bits {};
    struct cmp_gt {};
    struct cmp_lt {};
    struct cmp_eq {};
    struct cmp_ge {};
    template<int LANE> struct bcast_lane {};
}

/// Reads the item size back out of a mask type. Needed because the flavour a comparison returns
/// depends on the rank that was selected, and the caller has to key the next operation on it.
template<class M> struct mask_item_size;
template<int N,int IS,class Arch>
struct mask_item_size<internal::SimdMaskImpl<N,IS,Arch>> { static constexpr int value = IS; };

/// What a variant is chosen on. `MASK_BITS` is the mask item size for mask-taking operations
/// (1 = bits, 32 = full lanes) and stays 0 for the others.
template<class T,int N,class Arch,int MASK_BITS = 0> struct Key {};

// =====================================================================================
// GENERIC VARIANTS -- rank 0. Always available: this is what guarantees a fallback.
// =====================================================================================

/// THROUGH `mul` AND `add`, not through a lane loop. There is no fused multiply-add on any x86
/// integer type, so this generic form is what every `fma` on an integer lands on -- and written
/// as a loop over `values` it was a scalar loop unless the compiler happened to offer vector
/// arithmetic. `mul` and `add` have register forms; use them, and the fallback inherits them.
///
/// It costs nothing on floating point: where a real `vfmadd` exists the REGISTER variant wins,
/// and where it does not, a multiply followed by an add is exactly what the hardware would do.
template<class T,int N,class Arch>
struct sel::Variant<ops::fma,Key<T,N,Arch>,sel::GENERIC> {
    static constexpr bool available = true;
    using V = internal::SimdVecImpl<T,N,Arch>;
    static V run( const V &a, const V &b, const V &c ) {
        return internal::add( internal::mul( a, b ), c );
    }
};

template<class T,int N,class Arch>
struct sel::Variant<ops::permute,Key<T,N,Arch>,sel::GENERIC> {
    static constexpr bool available = true;
    using V = internal::SimdVecImpl<T,N,Arch>;
    using I = internal::SimdVecImpl<SI32,N,Arch>;
    static V run( const V &v, const I &idx ) {
        T tmp[ N ];
        for ( int i = 0; i < N; ++i ) tmp[ i ] = v.data.values[ i ];
        V res;
        // A MODULO, AND AN UNSIGNED ONE.
        //
        // `% N` and not `& ( N - 1 )`, because the mask is a modulo only at a power of two and
        // arbitrary widths are what asimd is for: at N = 5 the mask was `& 4`, so reversing five
        // lanes returned `50 10 10 10 10`.
        //
        // UNSIGNED, because that is what the hardware does. `vpermps` keeps the low three bits of
        // each index, so index -1 selects lane 7 -- and ARM's `TBL` control is masked the same
        // way, so the two backends agree without either being told about the other. The signed
        // `j < 0 ? j + N : j` written here sent it to lane 7 as well at N = 8 -- by coincidence
        // -- but to a different lane at any other width, so the generic and register forms
        // disagreed. Reading the index as unsigned makes them agree at every power of two, and is
        // one instruction cheaper: at a constant power-of-two N the compiler turns `% N` back
        // into the `and`.
        for ( int i = 0; i < N; ++i )
            res.data.values[ i ] = tmp[ PI32( idx.data.values[ i ] ) % PI32( N ) ];
        return res;
    }
};

template<int LANE,class T,int N,class Arch>
struct sel::Variant<ops::bcast_lane<LANE>,Key<T,N,Arch>,sel::GENERIC> {
    static constexpr bool available = true;
    using V = internal::SimdVecImpl<T,N,Arch>;
    static V run( const V &v ) {
        V res;
        for ( int i = 0; i < N; ++i ) res.data.values[ i ] = v.data.values[ LANE ];
        return res;
    }
};

template<class T,int N,class Arch,int IS>
struct sel::Variant<ops::select,Key<T,N,Arch,IS>,sel::GENERIC> {
    static constexpr bool available = true;
    using V = internal::SimdVecImpl<T,N,Arch>;
    using M = internal::SimdMaskImpl<N,IS,Arch>;
    static V run( const M &m, const V &a, const V &b ) {
        V res;
        for ( int i = 0; i < N; ++i )
            res.data.values[ i ] = m.data.values[ i ] ? a.data.values[ i ] : b.data.values[ i ];
        return res;
    }
};

/// `to_bits` returns a PI64, not an `unsigned`. A mask can be 64 lanes wide (`SI8` on
/// AVX-512BW), and `1u << i` is undefined for i >= 32 -- so the generic form was undefined
/// behaviour on exactly the widths that need it most.
template<int N,class Arch,int IS>
struct sel::Variant<ops::to_bits,Key<void,N,Arch,IS>,sel::GENERIC> {
    static constexpr bool available = true;
    using M = internal::SimdMaskImpl<N,IS,Arch>;
    static PI64 run( const M &m ) {
        PI64 res = 0;
        for ( int i = 0; i < N; ++i )
            res |= PI64( bool( m.data.values[ i ] ) ) << i;
        return res;
    }
};

template<int N,class Arch>
struct sel::Variant<ops::mask_from_bits,Key<void,N,Arch>,sel::GENERIC> {
    static constexpr bool available = true;
    static internal::SimdMaskImpl<N,32,Arch> run( PI64 b ) {
        internal::SimdMaskImpl<N,32,Arch> res;
        for ( int i = 0; i < N; ++i )
            res.data.values[ i ] = ( b >> i ) & 1 ? ~PI32( 0 ) : PI32( 0 );
        return res;
    }
};

/// The four comparisons. asimd only had `lt` and `gt`, and only as lazy expressions. `eq` is
/// needed to DESIGNATE A LANE (`iota == i` is the portable way of saying "lane i", where
/// intrinsics take an immediate mask) and `ge` to bring indices back into range.
#define ASIMD_PLUS_GENERIC_CMP( TAG, OP )                                                       \
    template<class T,int N,class Arch>                                                          \
    struct sel::Variant<ops::TAG,Key<T,N,Arch>,sel::GENERIC> {                                  \
        static constexpr bool available = true;                                                 \
        using V = internal::SimdVecImpl<T,N,Arch>;                                              \
        static internal::SimdMaskImpl<N,32,Arch> run( const V &a, const V &b ) {                \
            internal::SimdMaskImpl<N,32,Arch> res;                                              \
            for ( int i = 0; i < N; ++i )                                                       \
                res.data.values[ i ] = ( a.data.values[ i ] OP b.data.values[ i ] ) ? ~PI32( 0 ) \
                                                                                    : PI32( 0 ); \
            return res;                                                                         \
        }                                                                                       \
    }

ASIMD_PLUS_GENERIC_CMP( cmp_gt, >  );
ASIMD_PLUS_GENERIC_CMP( cmp_lt, <  );
ASIMD_PLUS_GENERIC_CMP( cmp_eq, == );
ASIMD_PLUS_GENERIC_CMP( cmp_ge, >= );

#undef ASIMD_PLUS_GENERIC_CMP

} // namespace asimd

// Register-level variants. Class specializations are looked up at instantiation, so this include
// could sit anywhere -- it is here for readability, not out of necessity. That is precisely what
// `Selection.h` buys: with qualified calls to overloaded functions, putting it after the facades
// below silently cost every bit of vectorization.
#include "SimdOpsPlus_X86.h"
#include "SimdOpsPlus_Neon.h"

// ... and the SPLIT-rank forms, which delegate to whatever the register level above resolved to.
// They have to come after it: `available` asks the halves what rank they reached. On ARM that is
// not a refinement but the main path: the register is 128 bits and never wider, so any width
// above four floats IS a split -- see the header comment of `SimdOpsPlus_Neon.h`.
#include "SimdOpsPlus_Split.h"

namespace asimd {

// =====================================================================================
// FACADES
// =====================================================================================

template<class T,int N,class Arch>
SimdVec<T,N,Arch> fma( const SimdVec<T,N,Arch> &a, const SimdVec<T,N,Arch> &b,
                       const SimdVec<T,N,Arch> &c ) {
    return sel::call<ops::fma,Key<T,N,Arch>>( a.impl, b.impl, c.impl );
}

template<class T,int N,class Arch>
SimdVec<T,N,Arch> permute( const SimdVec<T,N,Arch> &v, const SimdVec<SI32,N,Arch> &idx ) {
    return sel::call<ops::permute,Key<T,N,Arch>>( v.impl, idx.impl );
}

template<int LANE,class T,int N,class Arch>
SimdVec<T,N,Arch> bcast_lane( const SimdVec<T,N,Arch> &v ) {
    return sel::call<ops::bcast_lane<LANE>,Key<T,N,Arch>>( v.impl );
}

template<class T,int N,int IS,class Arch>
SimdVec<T,N,Arch> select( const SimdMask<N,IS,Arch> &m, const SimdVec<T,N,Arch> &a,
                          const SimdVec<T,N,Arch> &b ) {
    return sel::call<ops::select,Key<T,N,Arch,IS>>( m.impl, a.impl, b.impl );
}

template<int N,int IS,class Arch>
PI64 to_bits( const SimdMask<N,IS,Arch> &m ) {
    return sel::call<ops::to_bits,Key<void,N,Arch,IS>>( m.impl );
}

/// THE DUAL OF `to_bits`, and it is worth having. Designating a lane through `eq( iota, i )`
/// takes a `vpbroadcastd` then a `vpcmpeqd`; on a target with mask registers the same mask is one
/// `kmovb` from an integer you already computed. Code that designates a few lanes per iteration
/// pays that difference -- six instructions against three -- in its hottest loop.
///
/// It is also what lets portable code express "the first n lanes", "every other lane", or any
/// computed pattern -- all things a vector comparison can only say by abusing an `iota`.
template<int N,class Arch = NativeCpu>
auto mask_from_bits( PI64 b ) {
    return simd_mask_from_simd_mask_impl( sel::call<ops::mask_from_bits,Key<void,N,Arch>>( b ) );
}

/// COMPARISONS ARE LAZY IN ASIMD, and it is a good choice: `a > b` computes nothing, it returns
/// an object holding both operands, and the caller decides in which form to materialize it. These
/// facades are the two forms that were missing: "as bits" and "as a selection".
///
/// The return type is deduced on purpose: the mask FLAVOUR depends on the target -- bits where
/// there are mask registers, full lanes elsewhere -- and pinning it here would force a choice.
/// `select` and `to_bits` accept both.
#define ASIMD_PLUS_CMP_FACADE( NAME, TAG )                                                  \
    template<class T,int N,class Arch>                                                      \
    auto NAME( const SimdVec<T,N,Arch> &a, const SimdVec<T,N,Arch> &b ) {                   \
        return simd_mask_from_simd_mask_impl( sel::call<ops::TAG,Key<T,N,Arch>>( a.impl, b.impl ) ); \
    }

ASIMD_PLUS_CMP_FACADE( gt, cmp_gt )
ASIMD_PLUS_CMP_FACADE( lt, cmp_lt )
ASIMD_PLUS_CMP_FACADE( eq, cmp_eq )
ASIMD_PLUS_CMP_FACADE( ge, cmp_ge )

#undef ASIMD_PLUS_CMP_FACADE

/// `a > b` straight to bits, without naming the mask flavour in between.
template<class T,int N,class Arch>
PI64 to_bits( const internal::Op_gt<T,N,Arch> &op ) {
    auto m = sel::call<ops::cmp_gt,Key<T,N,Arch>>( op.a, op.b );
    return sel::call<ops::to_bits,Key<void,N,Arch,mask_item_size<decltype( m )>::value>>( m );
}
template<class T,int N,class Arch>
PI64 to_bits( const internal::Op_lt<T,N,Arch> &op ) {
    auto m = sel::call<ops::cmp_lt,Key<T,N,Arch>>( op.a, op.b );
    return sel::call<ops::to_bits,Key<void,N,Arch,mask_item_size<decltype( m )>::value>>( m );
}

/// idem for a selection driven by a lazy comparison.
template<class T,class U,int N,class Arch>
SimdVec<U,N,Arch> select( const internal::Op_gt<T,N,Arch> &op, const SimdVec<U,N,Arch> &a,
                          const SimdVec<U,N,Arch> &b ) {
    auto m = sel::call<ops::cmp_gt,Key<T,N,Arch>>( op.a, op.b );
    return sel::call<ops::select,Key<U,N,Arch,mask_item_size<decltype( m )>::value>>( m, a.impl, b.impl );
}

/// The operators that were missing -- the underlying operations already existed.
template<class T,int N,class Arch>
SimdVec<T,N,Arch> operator-( const SimdVec<T,N,Arch> &a, const SimdVec<T,N,Arch> &b ) {
    return internal::sub( a.impl, b.impl );
}
template<class T,int N,class Arch>
SimdVec<T,N,Arch> operator*( const SimdVec<T,N,Arch> &a, const SimdVec<T,N,Arch> &b ) {
    return internal::mul( a.impl, b.impl );
}
template<class T,int N,class Arch>
SimdVec<T,N,Arch> operator/( const SimdVec<T,N,Arch> &a, const SimdVec<T,N,Arch> &b ) {
    return internal::div( a.impl, b.impl );
}

} // namespace asimd
