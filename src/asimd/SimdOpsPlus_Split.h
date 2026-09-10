#pragma once

// =============================================================================================
// THE SPLIT RANK, which was declared in `Selection.h` from the start and never used.
//
// asimd's premise is that the width is the author's choice: `SimdVec<float,8>` must work on a
// target whose registers hold four. The vector impls do that -- they split recursively, and the
// arithmetic follows the split. The OPERATIONS ADDED BY `SimdOpsPlus.h` did not: their generic
// forms walk `values` lane by lane, ignoring the split entirely.
//
// Measured, `to_bits( a > b )` on `float x 8` under SSE2:
//
//     written with plain gcc vector extensions      48 instructions
//     asimd, generic forms                          73          <- worse than doing nothing
//     asimd, with the split forms below              9
//
// Being beaten by the compiler at the one thing the library exists for is the kind of result that
// makes a library not worth its include. The fix is not per-type work: a split form is the same
// three lines for every type and every width, because it delegates to whatever the halves resolve
// to -- a register form, a mask register form, or another split.
//
// `permute` is the awkward one, and it is included anyway. A permutation moves a lane from one
// half to the OTHER, which is exactly what two half-registers cannot do directly -- so it costs
// two half-permutations and a blend per half, four and two for the whole, where a target with a
// full-width `vpermps` needs one instruction. It is still worth having: the alternative is the
// generic form, which stores the vector to the stack, indexes it there and reloads. Slow beats a
// round trip through memory, and more to the point the contract is that EVERY operation works at
// EVERY width -- an operation that opts out of the split mechanism is a hole in the premise.
// =============================================================================================

#include "Selection.h"

namespace asimd {

namespace internal {

/// The two halves of an impl, or zero when it does not split. Written as a specialisation rather
/// than an `if constexpr` because `SimdVecImpl<T,1,Arch>` has no `split_size_0` to name at all.
template<class T,int N,class Arch,bool = HasSplit<SimdVecImpl<T,N,Arch>>>
struct SplitOf { static constexpr int n0 = 0, n1 = 0; };

template<class T,int N,class Arch>
struct SplitOf<T,N,Arch,true> {
    static constexpr int n0 = SimdVecImpl<T,N,Arch>::split_size_0;
    static constexpr int n1 = SimdVecImpl<T,N,Arch>::split_size_1;
};

/// idem for a mask.
template<int N,int IS,class Arch,bool = requires ( SimdMaskImpl<N,IS,Arch> m ) { m.data.split.v0; }>
struct MaskSplits { static constexpr bool value = false; };
template<int N,int IS,class Arch>
struct MaskSplits<N,IS,Arch,true> { static constexpr bool value = true; };

} // namespace internal

/// "is the half worth delegating to" -- i.e. does it resolve to something better than another
/// lane loop. Guarded by a specialisation so that `rank<..., Key<T,0,Arch>>` is never named.
template<class Op,class T,int N,class Arch,bool = ( N > 0 )>
struct HalfIsWorthIt { static constexpr bool value = false; };
template<class Op,class T,int N,class Arch>
struct HalfIsWorthIt<Op,T,N,Arch,true> {
    static constexpr bool value = sel::rank<Op,Key<T,N,Arch>> > sel::GENERIC;
};

/// what a comparison actually returns at a given width, once the selector has had its say. The
/// flavour is not fixed -- lane mask below AVX-512, bit mask above -- so it has to be read back.
template<class Op,class T,int N,class Arch>
using CmpResultOf = decltype( sel::call<Op,Key<T,N,Arch>>(
    std::declval<const internal::SimdVecImpl<T,N,Arch> &>(),
    std::declval<const internal::SimdVecImpl<T,N,Arch> &>() ) );

// ---------------------------------------------------------------------------------------------
// comparisons
// ---------------------------------------------------------------------------------------------
#define ASIMD_PLUS_SPLIT_CMP( TAG )                                                              \
    template<class T,int N,class Arch>                                                           \
    struct sel::Variant<ops::TAG,Key<T,N,Arch>,sel::SPLIT> {                                     \
        using V = internal::SimdVecImpl<T,N,Arch>;                                               \
        static constexpr int n0 = internal::SplitOf<T,N,Arch>::n0;                               \
        static constexpr int n1 = internal::SplitOf<T,N,Arch>::n1;                               \
        static constexpr int is0 = ASIMD_ITEM_SIZE_OR_0( TAG, n0 );                              \
        static constexpr int is1 = ASIMD_ITEM_SIZE_OR_0( TAG, n1 );                              \
        /* both halves must agree on the mask flavour, and the whole must be able to hold two */ \
        static constexpr bool available =                                                        \
            HalfIsWorthIt<ops::TAG,T,n0,Arch>::value && is0 != 0 && is0 == is1                   \
            && internal::MaskSplits<N,is0,Arch>::value;                                          \
        static internal::SimdMaskImpl<N,is0,Arch> run( const V &a, const V &b ) {                \
            internal::SimdMaskImpl<N,is0,Arch> res;                                              \
            res.data.split.v0 = sel::call<ops::TAG,Key<T,n0,Arch>>( a.data.split.v0, b.data.split.v0 ); \
            res.data.split.v1 = sel::call<ops::TAG,Key<T,n1,Arch>>( a.data.split.v1, b.data.split.v1 ); \
            return res;                                                                          \
        }                                                                                        \
    }

/// the mask item size a comparison yields at width N, or 0 when N is not a real width.
template<class Op,class T,int N,class Arch,bool = ( N > 0 )>
struct ItemSizeOf { static constexpr int value = 0; };
template<class Op,class T,int N,class Arch>
struct ItemSizeOf<Op,T,N,Arch,true> {
    static constexpr int value = mask_item_size<CmpResultOf<Op,T,N,Arch>>::value;
};

#define ASIMD_ITEM_SIZE_OR_0( TAG, NN ) ItemSizeOf<ops::TAG,T,NN,Arch>::value

ASIMD_PLUS_SPLIT_CMP( cmp_gt );
ASIMD_PLUS_SPLIT_CMP( cmp_lt );
ASIMD_PLUS_SPLIT_CMP( cmp_eq );
ASIMD_PLUS_SPLIT_CMP( cmp_ge );

#undef ASIMD_ITEM_SIZE_OR_0
#undef ASIMD_PLUS_SPLIT_CMP

// ---------------------------------------------------------------------------------------------
// to_bits -- the one that was 73 instructions. Each half yields its own bits; shift and or.
// ---------------------------------------------------------------------------------------------
template<int N,int IS,class Arch,bool = ( N > 0 )>
struct MaskHalfWorthIt { static constexpr bool value = false; };
template<int N,int IS,class Arch>
struct MaskHalfWorthIt<N,IS,Arch,true> {
    static constexpr bool value = sel::rank<ops::to_bits,Key<void,N,Arch,IS>> > sel::GENERIC;
};

template<int N,int IS,class Arch,bool = requires ( internal::SimdMaskImpl<N,IS,Arch> m ) { m.data.split.v0; }>
struct MaskSplitSizes { static constexpr int n0 = 0, n1 = 0; };
template<int N,int IS,class Arch>
struct MaskSplitSizes<N,IS,Arch,true> {
    static constexpr int n0 = internal::SimdMaskImpl<N,IS,Arch>::split_size_0;
    static constexpr int n1 = internal::SimdMaskImpl<N,IS,Arch>::split_size_1;
};

template<int N,class Arch,int IS>
struct sel::Variant<ops::to_bits,Key<void,N,Arch,IS>,sel::SPLIT> {
    static constexpr int n0 = MaskSplitSizes<N,IS,Arch>::n0;
    static constexpr int n1 = MaskSplitSizes<N,IS,Arch>::n1;
    static constexpr bool available = MaskHalfWorthIt<n0,IS,Arch>::value;
    static PI64 run( const internal::SimdMaskImpl<N,IS,Arch> &m ) {
        return sel::call<ops::to_bits,Key<void,n0,Arch,IS>>( m.data.split.v0 )
             | ( sel::call<ops::to_bits,Key<void,n1,Arch,IS>>( m.data.split.v1 ) << n0 );
    }
};

// ---------------------------------------------------------------------------------------------
// select -- two half blends
// ---------------------------------------------------------------------------------------------
template<class T,int N,class Arch,int IS,bool = ( N > 0 )>
struct SelectHalfWorthIt { static constexpr bool value = false; };
template<class T,int N,class Arch,int IS>
struct SelectHalfWorthIt<T,N,Arch,IS,true> {
    static constexpr bool value = sel::rank<ops::select,Key<T,N,Arch,IS>> > sel::GENERIC;
};

template<class T,int N,class Arch,int IS>
struct sel::Variant<ops::select,Key<T,N,Arch,IS>,sel::SPLIT> {
    using V = internal::SimdVecImpl<T,N,Arch>;
    using M = internal::SimdMaskImpl<N,IS,Arch>;
    static constexpr int n0 = internal::SplitOf<T,N,Arch>::n0;
    static constexpr int n1 = internal::SplitOf<T,N,Arch>::n1;
    static constexpr bool available = SelectHalfWorthIt<T,n0,Arch,IS>::value
        && internal::MaskSplits<N,IS,Arch>::value
        && MaskSplitSizes<N,IS,Arch>::n0 == n0;
    static V run( const M &m, const V &a, const V &b ) {
        V res;
        res.data.split.v0 = sel::call<ops::select,Key<T,n0,Arch,IS>>( m.data.split.v0, a.data.split.v0, b.data.split.v0 );
        res.data.split.v1 = sel::call<ops::select,Key<T,n1,Arch,IS>>( m.data.split.v1, a.data.split.v1, b.data.split.v1 );
        return res;
    }
};

// ---------------------------------------------------------------------------------------------
// fma -- three-operand, otherwise identical
// ---------------------------------------------------------------------------------------------
template<class T,int N,class Arch>
struct sel::Variant<ops::fma,Key<T,N,Arch>,sel::SPLIT> {
    using V = internal::SimdVecImpl<T,N,Arch>;
    static constexpr int n0 = internal::SplitOf<T,N,Arch>::n0;
    static constexpr int n1 = internal::SplitOf<T,N,Arch>::n1;
    static constexpr bool available = HalfIsWorthIt<ops::fma,T,n0,Arch>::value;
    static V run( const V &a, const V &b, const V &c ) {
        V res;
        res.data.split.v0 = sel::call<ops::fma,Key<T,n0,Arch>>( a.data.split.v0, b.data.split.v0, c.data.split.v0 );
        res.data.split.v1 = sel::call<ops::fma,Key<T,n1,Arch>>( a.data.split.v1, b.data.split.v1, c.data.split.v1 );
        return res;
    }
};

// ---------------------------------------------------------------------------------------------
// permute -- the cross-half one
//
//   r_k = select( i_k < n0, permute( v0, i_k ), permute( v1, i_k - n0 ) )
//
// for each half k of the index. Both half-permutations are computed and one is thrown away: a
// permutation index that points into the other half is out of range for the half we hand it to,
// which is harmless because the sub-permute wraps it (`vpermps` keeps the low bits, the generic
// form takes a modulo) and the blend discards that lane anyway.
//
// ONLY WHEN THE TWO HALVES HAVE THE SAME WIDTH. At a width that is not a power of two the halves
// differ -- 5 splits into 4 + 1 -- and there is no type in which to express "permute a one-lane
// vector with a four-lane index". Those widths keep the generic form, which is correct.
// ---------------------------------------------------------------------------------------------
template<class T,int N,class Arch>
struct sel::Variant<ops::permute,Key<T,N,Arch>,sel::SPLIT> {
    using V  = internal::SimdVecImpl<T,N,Arch>;
    using I  = internal::SimdVecImpl<SI32,N,Arch>;
    static constexpr int n0 = internal::SplitOf<T,N,Arch>::n0;
    static constexpr int n1 = internal::SplitOf<T,N,Arch>::n1;
    using Vh = internal::SimdVecImpl<T,n0,Arch>;
    using Ih = internal::SimdVecImpl<SI32,n0,Arch>;
    static constexpr int is = ItemSizeOf<ops::cmp_lt,SI32,n0,Arch>::value;

    /// ALL THREE PIECES MUST BE REGISTER FORMS, not just the permutation.
    ///
    /// A split permutation is four half-permutations plus two blends. If the BLEND is a lane
    /// loop, those two dominate and the whole thing loses to the generic form -- which stores to
    /// the stack and reloads, and is at least linear. Measured under -mssse3, where `pshufb`
    /// gives a permutation at four lanes but `blendv` needs SSE4.1: the split form came out at
    /// 75 instructions against the generic form's 51, and was selected anyway because its rank
    /// was higher.
    ///
    /// A rank says "better if available". Deciding whether it IS better is what `available` is
    /// for, and for a composite form that means asking about every piece.
    static constexpr bool available =
        n0 > 0 && n0 == n1
        && HalfIsWorthIt<ops::permute,T,n0,Arch>::value
        && SelectHalfWorthIt<T,n0,Arch,is>::value;

    /// the index vector does not necessarily split the way the value vector does: on AVX2,
    /// `SimdVec<double,8>` is two registers while `SimdVec<SI32,8>` is one. So take the halves
    /// through `values` when there is no split to take them from.
    template<int OFF>
    static Ih idx_half( const I &idx ) {
        if constexpr ( internal::HasSplit<I> && I::split_size_0 == n0 ) {
            if constexpr ( OFF == 0 ) return idx.data.split.v0; else return idx.data.split.v1;
        } else {
            Ih r;
            for ( int i = 0; i < n0; ++i ) r.data.values[ i ] = idx.data.values[ OFF + i ];
            return r;
        }
    }

    static Vh splat_n0() { Ih r; internal::init_sc( r, SI32( n0 ) ); return r; }

    static Vh half( const Vh &v0, const Vh &v1, const Ih &ih ) {
        Ih lo; internal::init_sc( lo, SI32( n0 ) );
        const Vh from_0 = sel::call<ops::permute,Key<T,n0,Arch>>( v0, ih );
        const Vh from_1 = sel::call<ops::permute,Key<T,n0,Arch>>( v1, internal::sub( ih, lo ) );
        const auto m    = sel::call<ops::cmp_lt,Key<SI32,n0,Arch>>( ih, lo );
        return sel::call<ops::select,Key<T,n0,Arch,is>>( m, from_0, from_1 );
    }

    static V run( const V &v, const I &idx ) {
        V res;
        res.data.split.v0 = half( v.data.split.v0, v.data.split.v1, idx_half<0>( idx ) );
        res.data.split.v1 = half( v.data.split.v0, v.data.split.v1, idx_half<n0>( idx ) );
        return res;
    }
};

} // namespace asimd
