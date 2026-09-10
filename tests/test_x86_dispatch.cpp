// WHICH VARIANT ACTUALLY GETS PICKED, over the whole grid -- not at the one width the porting
// work happened to need.
//
// `test_selection.cpp` checks that nothing silently drops to the generic form, and it checks it
// for `Key<float,8,NativeCpu>`. That is exactly the cell `SimdOpsPlus_X86.h` implements, so the
// safety net can only ever say yes. This file asks the same question at every width and every
// type, and in particular AT THE NATIVE WIDTH -- the width `SimdVec<T>` picks when you do not
// name one, which on an AVX-512 machine is 16 floats, not 8.
//
// A rank of 0 (GENERIC) is not a wrong answer: it is a correct answer computed lane by lane. It
// is a PERFORMANCE bug, and the kind that no value-comparing test can see. That is the whole
// argument of `Selection.h`, applied to itself.
//
// The `require_at_least` calls below are the durable half of this file: they are static_asserts,
// so a lost registration stops the BUILD instead of quietly costing a factor of two.

#include <asimd/SimdOpsPlus.h>

#include "check.h"

using namespace asimd;
using A = NativeCpu;

static int nb_generic = 0, nb_cells = 0;

/// prints one cell and counts it. `want` says whether this cell is expected to have a register
/// form on this build -- a 4-lane float has one only if the target has SSE, and so on.
template<class Op,class T,int N>
static void cell() {
    constexpr int r = sel::rank<Op,Key<T,N,A>>;
    ++nb_cells;
    if ( r == sel::GENERIC ) ++nb_generic;
    printf( "%6d", r );
}

template<class Op>
static void row( const char *name ) {
    printf( "  %-16s", name );
    cell<Op,float ,4>(); cell<Op,float ,8>(); cell<Op,float ,16>();
    cell<Op,double,2>(); cell<Op,double,4>(); cell<Op,double, 8>();
    cell<Op,SI32  ,4>(); cell<Op,SI32  ,8>(); cell<Op,SI32  ,16>();
    printf( "\n" );
}

/// the rank at the width `SimdVec<T>` picks on its own.
template<class Op,class T>
static int rank_at_native_width() { return sel::rank<Op,Key<T,SimdSize<T,A>::value,A>>; }

/// WHAT MUST NOT REGRESS. `require_at_least` turns a silent fallback into a compile error, and it
/// is the whole point of `Selection.h` -- it was simply never called outside a demonstration.
/// These are static_asserts: if a registration is lost, or a feature guard is widened by mistake,
/// the BUILD stops here rather than the benchmark noticing months later.
///
/// A TEMPLATE, and that is not decoration. `if constexpr` discards the untaken branch only inside
/// a template: in a plain function both branches are compiled, so every one of these assertions
/// fired regardless of the target. gcc let it pass, clang did not -- and clang is right.
/// Parameterising on the architecture is what makes the guards actually guard.
template<class A>
static void require_the_floor() {
    // Conditional on the target actually having the instructions, so the same file is
    // meaningful under -msse2, -mavx, -mavx2 and -march=native.
    if constexpr ( A::template Has<features::SSE2>::value ) {
        sel::require_at_least<ops::cmp_gt,Key<float,4,A>,sel::REGISTER>();
        sel::require_at_least<ops::cmp_eq,Key<SI32 ,4,A>,sel::REGISTER>();
        sel::require_at_least<ops::to_bits,Key<void,4,A,32>,sel::REGISTER>();
        sel::require_at_least<ops::bcast_lane<1>,Key<float,4,A>,sel::REGISTER>();
    }
    if constexpr ( A::template Has<features::SSE4_1>::value )
        sel::require_at_least<ops::select,Key<float,4,A,32>,sel::REGISTER>();
    if constexpr ( A::template Has<features::AVX>::value ) {
        sel::require_at_least<ops::cmp_gt,Key<float ,8,A>,sel::REGISTER>();
        sel::require_at_least<ops::cmp_gt,Key<double,4,A>,sel::REGISTER>();
        sel::require_at_least<ops::select,Key<float ,8,A,32>,sel::REGISTER>();
        sel::require_at_least<ops::permute,Key<float,4,A>,sel::REGISTER>();
    }
    if constexpr ( A::template Has<features::AVX2>::value ) {
        sel::require_at_least<ops::permute,Key<float,8,A>,sel::REGISTER>();
        sel::require_at_least<ops::cmp_gt,Key<SI32,8,A>,sel::REGISTER>();
        sel::require_at_least<ops::bcast_lane<1>,Key<float,8,A>,sel::REGISTER>();
    }
    // `Has<FMA>` ALONE IS NOT AN x86 TEST, and this block used to be written as though it were.
    // `features::FMA` is shared with ARM -- it says the target can fuse, not which instruction
    // does it (see GenericFeatures.h) -- so on an AArch64 build these three fired and demanded
    // `vfmadd`-class ranks at x86 widths. It has to be conjoined with the feature that gives the
    // WIDTH, which is also the more accurate statement: FMA without AVX cannot fuse eight lanes.
    if constexpr ( A::template Has<features::FMA>::value && A::template Has<features::SSE2>::value )
        sel::require_at_least<ops::fma,Key<float ,4,A>,sel::REGISTER>();
    if constexpr ( A::template Has<features::FMA>::value && A::template Has<features::AVX>::value ) {
        sel::require_at_least<ops::fma,Key<float ,8,A>,sel::REGISTER>();
        sel::require_at_least<ops::fma,Key<double,4,A>,sel::REGISTER>();
    }
    if constexpr ( A::template Has<features::AVX512>::value ) {
        // the widths `SimdVec<T>` picks on its own on this target
        sel::require_at_least<ops::fma    ,Key<float ,16,A>,sel::REGISTER>();
        sel::require_at_least<ops::fma    ,Key<double, 8,A>,sel::REGISTER>();
        sel::require_at_least<ops::permute,Key<float ,16,A>,sel::REGISTER>();
        sel::require_at_least<ops::permute,Key<double, 8,A>,sel::REGISTER>();
        sel::require_at_least<ops::cmp_gt ,Key<float ,16,A>,sel::MASK_REGISTER>();
        sel::require_at_least<ops::cmp_gt ,Key<PI32  ,16,A>,sel::MASK_REGISTER>();
        sel::require_at_least<ops::select ,Key<float ,16,A,1>,sel::MASK_REGISTER>();
        sel::require_at_least<ops::to_bits,Key<void  ,16,A,1>,sel::MASK_REGISTER>();
    }
    // ---- THE SPLIT FLOOR: widths above the register ----------------------------------------
    //
    // Every width that does not fit one register has to reach SPLIT, or the library is walking
    // lanes for a value that fits in two registers. This was never asserted on x86 -- the ABI
    // probe was standing in for it, by noticing that a lane loop spills, and that turned out to
    // be the wrong instrument: at those widths the value goes through memory whatever the
    // dispatch does (SysV cannot pass 64 bytes in registers without AVX-512), so the probe was
    // asserting something no code change could deliver. A rank floor is the right tool, and it
    // is a static_assert.
    if constexpr ( A::template Has<features::SSE2>::value && ! A::template Has<features::AVX>::value )
        // eight floats are two `xmm` here
        sel::require_at_least<ops::cmp_gt,Key<float,8,A>,sel::SPLIT>();
    if constexpr ( A::template Has<features::AVX>::value && ! A::template Has<features::AVX512>::value ) {
        // ... and sixteen are two `ymm`, as are eight doubles
        sel::require_at_least<ops::cmp_gt,Key<float ,16,A>,sel::SPLIT>();
        sel::require_at_least<ops::select,Key<float ,16,A,32>,sel::SPLIT>();
    }

    // `fma` IS ONLY EXPECTED TO SPLIT WHERE THERE IS SOMETHING TO SPLIT INTO. Written without
    // the FMA condition, these fired on every target that has no `vfmadd` at all -- and they
    // were wrong to: with no register `fma` at any width the split form correctly reports itself
    // unavailable, and the GENERIC form is `add( mul( a, b ), c )`, whose `mul` and `add` have
    // register forms and split on their own. Rank 0 there is two `mulps` and two `addps`, not a
    // lane loop. The floor has to ask for what the target can actually give.
    if constexpr ( A::template Has<features::FMA>::value && A::template Has<features::SSE2>::value
                && ! A::template Has<features::AVX>::value )
        sel::require_at_least<ops::fma,Key<float,8,A>,sel::SPLIT>();
    if constexpr ( A::template Has<features::FMA>::value && A::template Has<features::AVX>::value
                && ! A::template Has<features::AVX512>::value ) {
        sel::require_at_least<ops::fma,Key<float ,16,A>,sel::SPLIT>();
        // the cell the ABI probe used to be watching, asserted properly: `fma` on eight doubles
        // was a generic LANE LOOP once, and the probe reported it as 23 instructions with 12
        // stack accesses. This says the same thing by name, and without depending on an ABI that
        // cannot pass 64 bytes in registers anyway.
        sel::require_at_least<ops::fma,Key<double, 8,A>,sel::SPLIT>();
    }

    if constexpr ( A::template Has<features::AVX512VL>::value ) {
        // AVX-512VL is what puts the MASK REGISTERS on the narrower widths. Without these a
        // comparison at eight lanes still produced a 256-bit lane mask and `select` still
        // emitted `vblendvps`.
        sel::require_at_least<ops::cmp_gt,Key<float,8,A>,sel::MASK_REGISTER>();
        sel::require_at_least<ops::cmp_gt,Key<float,4,A>,sel::MASK_REGISTER>();
        sel::require_at_least<ops::select,Key<float,8,A,1>,sel::MASK_REGISTER>();
        sel::require_at_least<ops::permute,Key<double,4,A>,sel::REGISTER>();
    }
}

int main() {
    printf( "  architecture   : %s\n", A::name().c_str() );
    printf( "  native widths  : float=%d double=%d SI32=%d\n\n",
            SimdSize<float,A>::value, SimdSize<double,A>::value, SimdSize<SI32,A>::value );

    printf( "  %-16s%6s%6s%6s%6s%6s%6s%6s%6s%6s\n", "op \\ (type,lanes)",
            "f,4", "f,8", "f,16", "d,2", "d,4", "d,8", "i,4", "i,8", "i,16" );
    row<ops::fma>           ( "fma"           );
    row<ops::permute>       ( "permute"       );
    row<ops::bcast_lane<1>> ( "bcast_lane<1>" );
    row<ops::cmp_gt>        ( "cmp_gt"        );
    row<ops::cmp_lt>        ( "cmp_lt"        );
    row<ops::cmp_eq>        ( "cmp_eq"        );
    row<ops::cmp_ge>        ( "cmp_ge"        );
    printf( "  (0 = GENERIC, a scalar loop; %d = SPLIT; %d = REGISTER; %d = MASK_REGISTER)\n\n",
            sel::SPLIT, sel::REGISTER, sel::MASK_REGISTER );

    require_the_floor<A>();

    // ---- THE WIDTH YOU GET WHEN YOU DO NOT NAME ONE. -----------------------------------------
    //
    // `SimdVec<float>` is `SimdVec<float,SimdSize<float>::value>`: 16 lanes on an AVX-512 target,
    // 8 on AVX2, 4 on SSE2. This file used to register nothing but 8, so the default vector type
    // got the SLOWEST path on the WIDEST machine -- `fma` at 16 lanes measured 23 instructions
    // with 12 stack accesses, against 3 and 0 at 8 lanes.
    const int r_fma  = rank_at_native_width<ops::fma,    float>();
    const int r_perm = rank_at_native_width<ops::permute,float>();
    const int r_gt   = rank_at_native_width<ops::cmp_gt, float>();
    printf( "  at the native width, SimdVec<float> gets: fma=%d permute=%d cmp_gt=%d\n",
            r_fma, r_perm, r_gt );
    if constexpr ( A::template Has<features::FMA>::value && A::template Has<features::SSE2>::value ) CHECK( r_fma > sel::GENERIC );
    if constexpr ( A::template Has<features::SSE2>::value ) CHECK( r_gt > sel::GENERIC );
    // a permutation ACROSS a whole register is `vpermilps` at 128 bits (AVX), `vpermps` at 256
    // (AVX2), `vpermps`/`vpermpd` at 512. So the native width has one as soon as there is AVX2 --
    // and under -mavx alone the native width is already 8, where `vpermilps` cannot reach.
    if constexpr ( A::template Has<features::AVX2>::value ) CHECK( r_perm > sel::GENERIC );
    else if constexpr ( A::template Has<features::AVX>::value && SimdSize<float,A>::value == 4 )
        CHECK( r_perm > sel::GENERIC );

    // ---- what is LEGITIMATELY generic, and nothing else --------------------------------------
    //
    // Two things in the grid above have no instruction to map onto, at any width:
    //   - `fma` on the integer types: x86 has no integer fused multiply-add. The generic form
    //     is a multiply and an add, which is what the hardware would do anyway.
    //   - `permute` of two 64-bit lanes: the index vector would be `SI32 x 2`, a width with no
    //     register form of its own, and a two-lane permutation is two moves regardless.
    printf( "  %d of %d cells in the grid above are on the generic path.\n", nb_generic, nb_cells );
    if constexpr ( A::template Has<features::AVX512>::value )
        CHECK( nb_generic <= 4 );

    return report( "test_x86_dispatch" );
}
