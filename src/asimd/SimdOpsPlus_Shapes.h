#pragma once

// =============================================================================================
// THE SHAPE OF A REGISTER-LEVEL VARIANT -- one macro per (operation, mask flavour), and nothing
// architecture-specific in any of them.
//
// These lived in `SimdOpsPlus_X86.h`, which was fine while there was one backend. They are here
// because the ARM backend needs THE SAME ONES, exactly: what a variant of `select` looks like is
// a property of `Selection.h` and of `SimdMaskImpl`, not of the instruction set. Copying eighty
// lines into `SimdOpsPlus_Neon.h` would have meant two places to keep in step, and the shapes
// are the half least likely to differ -- a backend is a TABLE, and this is the table's header row.
//
// So a new backend now consists of `#include "SimdOpsPlus_Shapes.h"` plus its own rows. What
// each macro leaves to the row: `a`/`b` for a binary operation, `v` for a unary one, `m` for a
// mask, `idx` for a permutation index, `LANE` for a compile-time lane -- all named inside FUNC.
// =============================================================================================

#include "Selection.h"

namespace asimd {

/// "this target has that feature", and the two-feature conjunction. A vector width and the
/// instruction that uses it are often separate features -- FMA on x86, FMA on ARMv7 -- and the
/// conjunction is how a variant says it needs both.
#define ASIMD_PLUS_REQ1( C1 )     Arch::template Has<features::C1>::value
#define ASIMD_PLUS_REQ2( C1, C2 ) ( Arch::template Has<features::C1>::value && Arch::template Has<features::C2>::value )

/// "this feature, and NOT that one". Two backends at the same rank are ambiguous -- the rank
/// orders the levels, not the variants inside one (see the KNOWN LIMITS note in Selection.h).
/// Where an older instruction is a strictly worse fallback for a newer one, saying so in the
/// constraint is more honest than inventing a rank between REGISTER and REGISTER.
#define ASIMD_PLUS_REQ_EXCL( C1, CNOT ) ( Arch::template Has<features::C1>::value && ! Arch::template Has<features::CNOT>::value )

/// fma. Two features: the one that gives the width, and FMA itself -- they are orthogonal on
/// paper and were orthogonal in practice on the first AMD parts to carry FMA, and on ARMv7,
/// where fusing needs VFPv4.
#define ASIMD_PLUS_FMA( C1, C2, T, N, FUNC ) \
    template<class Arch> requires ( ASIMD_PLUS_REQ2( C1, C2 ) ) \
    struct sel::Variant<ops::fma,Key<T,N,Arch>,sel::REGISTER> { \
        static constexpr bool available = true; \
        using V = internal::SimdVecImpl<T,N,Arch>; \
        static V run( const V &a, const V &b, const V &c ) { \
            V res; res.data.reg = FUNC( a.data.reg, b.data.reg, c.data.reg ); return res; } \
    }

/// idem, but with the operands passed to FUNC in whatever order the instruction wants. ARM's
/// `vfmaq_f32( acc, x, y )` computes `acc + x * y`, so the ACCUMULATOR COMES FIRST -- the
/// opposite of `_mm_fmadd_ps( x, y, acc )`. A shape that fixed the order would force every ARM
/// row to wrap itself in a lambda.
#define ASIMD_PLUS_FMA_EXPR( C1, C2, T, N, FUNC ) \
    template<class Arch> requires ( ASIMD_PLUS_REQ2( C1, C2 ) ) \
    struct sel::Variant<ops::fma,Key<T,N,Arch>,sel::REGISTER> { \
        static constexpr bool available = true; \
        using V = internal::SimdVecImpl<T,N,Arch>; \
        static V run( const V &a, const V &b, const V &c ) { \
            V res; res.data.reg = FUNC; return res; } \
    }

/// variable-index permutation. `v` and the index vector may have different element types, so the
/// index impl is named separately.
#define ASIMD_PLUS_PERMUTE( C1, T, N, FUNC ) \
    template<class Arch> requires ( ASIMD_PLUS_REQ1( C1 ) ) \
    struct sel::Variant<ops::permute,Key<T,N,Arch>,sel::REGISTER> { \
        static constexpr bool available = true; \
        using V = internal::SimdVecImpl<T,N,Arch>; \
        using I = internal::SimdVecImpl<SI32,N,Arch>; \
        static V run( const V &v, const I &idx ) { V res; res.data.reg = FUNC; return res; } \
    }

/// `permute` with an explicit exclusion, for exactly the case ASIMD_PLUS_REQ_EXCL describes.
#define ASIMD_PLUS_PERMUTE_EXCL( C1, CNOT, T, N, FUNC ) \
    template<class Arch> requires ( ASIMD_PLUS_REQ_EXCL( C1, CNOT ) ) \
    struct sel::Variant<ops::permute,Key<T,N,Arch>,sel::REGISTER> { \
        static constexpr bool available = true; \
        using V = internal::SimdVecImpl<T,N,Arch>; \
        using I = internal::SimdVecImpl<SI32,N,Arch>; \
        static V run( const V &v, const I &idx ) { V res; res.data.reg = FUNC; return res; } \
    }

/// broadcast of a compile-time lane. `LANE` is available inside FUNC.
#define ASIMD_PLUS_BCAST( C1, T, N, FUNC ) \
    template<int LANE,class Arch> requires ( ASIMD_PLUS_REQ1( C1 ) ) \
    struct sel::Variant<ops::bcast_lane<LANE>,Key<T,N,Arch>,sel::REGISTER> { \
        static constexpr bool available = true; \
        using V = internal::SimdVecImpl<T,N,Arch>; \
        static V run( const V &v ) { V res; res.data.reg = FUNC; return res; } \
    }

/// comparison yielding a LANE mask (rank REGISTER) or a BIT mask (rank MASK_REGISTER).
#define ASIMD_PLUS_CMP( C1, TAG, T, N, IS, RANK, FUNC ) \
    template<class Arch> requires ( ASIMD_PLUS_REQ1( C1 ) ) \
    struct sel::Variant<ops::TAG,Key<T,N,Arch>,sel::RANK> { \
        static constexpr bool available = true; \
        using V = internal::SimdVecImpl<T,N,Arch>; \
        static internal::SimdMaskImpl<N,IS,Arch> run( const V &a, const V &b ) { \
            internal::SimdMaskImpl<N,IS,Arch> res; res.data.reg = FUNC; return res; } \
    }

/// blend. `m`, `a`, `b` are available inside FUNC; the result is `m ? a : b`.
#define ASIMD_PLUS_SELECT( C1, T, N, IS, RANK, FUNC ) \
    template<class Arch> requires ( ASIMD_PLUS_REQ1( C1 ) ) \
    struct sel::Variant<ops::select,Key<T,N,Arch,IS>,sel::RANK> { \
        static constexpr bool available = true; \
        using V = internal::SimdVecImpl<T,N,Arch>; \
        using M = internal::SimdMaskImpl<N,IS,Arch>; \
        static V run( const M &m, const V &a, const V &b ) { \
            V res; res.data.reg = FUNC; return res; } \
    }

#define ASIMD_PLUS_TO_BITS( C1, N, IS, RANK, FUNC ) \
    template<class Arch> requires ( ASIMD_PLUS_REQ1( C1 ) ) \
    struct sel::Variant<ops::to_bits,Key<void,N,Arch,IS>,sel::RANK> { \
        static constexpr bool available = true; \
        using M = internal::SimdMaskImpl<N,IS,Arch>; \
        static PI64 run( const M &m ) { return FUNC; } \
    }

#define ASIMD_PLUS_MASK_FROM_BITS( C1, N, IS, RANK, FUNC ) \
    template<class Arch> requires ( ASIMD_PLUS_REQ1( C1 ) ) \
    struct sel::Variant<ops::mask_from_bits,Key<void,N,Arch>,sel::RANK> { \
        static constexpr bool available = true; \
        static internal::SimdMaskImpl<N,IS,Arch> run( PI64 b ) { \
            internal::SimdMaskImpl<N,IS,Arch> res; res.data.reg = FUNC; return res; } \
    }

} // namespace asimd
