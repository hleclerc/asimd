// WHICH VARIANT ACTUALLY GETS PICKED ON ARM, over the whole grid -- the counterpart of
// `test_x86_dispatch.cpp`, and it has one more thing to prove.
//
// On x86 the interesting question was "did the registration for THIS width get written". The
// register grew from 128 to 512 bits, so most widths a caller asks for have a register of their
// own and the dispatch table is mostly a table.
//
// ON ARM THE REGISTER IS 128 BITS AND HAS BEEN SINCE 2005. `SimdVec<float,8>` -- the type the
// README opens with -- has no register on this architecture and never will: SVE's width is not a
// compile-time constant, so it cannot be a `SimdVecImpl` register at all. Which means the SPLIT
// rank is not a nicety here, it is HOW THE LIBRARY WORKS on ARM, and "did it reach SPLIT" is as
// important a question as "did it reach REGISTER". Both floors are asserted below.
//
// The `require_at_least` calls are the durable half of this file: they are static_asserts, so a
// lost registration stops the BUILD instead of quietly costing a factor of five.

#include <asimd/SimdOpsPlus.h>

#include "check.h"

using namespace asimd;
using A = NativeCpu;

// The ARMv7-A floor, on whatever machine this is compiled on: NEON without ASIMD. Everything
// registered against `ASIMD` -- `fdiv`, the reductions, `TBL`, the 64-bit compares, double
// precision -- has to disappear from under it and leave a correct answer behind. That is what
// the `X86Cpu<64,SSE2,SSE>` line does for the x86 tests, and it is the only way to exercise the
// 32-bit ARM lattice without a 32-bit ARM part.
using V7 = ArmCpu<64, features::NEON, features::FMA>;

static int nb_generic = 0, nb_cells = 0;

template<class Op,class T,int N,class Arch>
static void cell() {
    constexpr int r = sel::rank<Op,Key<T,N,Arch>>;
    ++nb_cells;
    if ( r == sel::GENERIC ) ++nb_generic;
    printf( "%6d", r );
}

template<class Op,class Arch>
static void row( const char *name ) {
    printf( "  %-16s", name );
    cell<Op,float ,4,Arch>(); cell<Op,float ,8,Arch>(); cell<Op,float ,16,Arch>();
    cell<Op,double,2,Arch>(); cell<Op,double,4,Arch>(); cell<Op,double, 8,Arch>();
    cell<Op,SI32  ,4,Arch>(); cell<Op,SI32  ,8,Arch>(); cell<Op,SI32  ,16,Arch>();
    printf( "\n" );
}

/// THE RANK OF ONE CELL, THROUGH A FUNCTION TEMPLATE -- and that is not a stylistic choice.
///
/// Naming `sel::rank<...>` DIRECTLY inside a non-template function is what this file did, and on
/// gcc 13 targeting x86 it produced:
///
///   Selection.h:94: error: 'constexpr int asimd::sel::search() [...with int R = 40]'
///                          used before its definition
///
/// on `cmp_gt`, `select` and `permute` at `Key<float,4,...>` -- exactly the three cells `main`
/// named. An instantiation-ordering complaint, on clang and gcc 15 invisible.
///
/// Two things changed because of it. `sel::search` is no longer a `constexpr` function template
/// at all -- see the long note in `Selection.h`, which is where the fix belongs, since the
/// diagnostic named a piece of the mechanism rather than of this test. And the ranks here are
/// materialised through these templates first, the way `test_x86_dispatch.cpp` has always done
/// it with `rank_at_native_width<Op,T>()`: inside a template the point of instantiation is well
/// defined, and the value arrives as a plain `int` that any later `if constexpr` can compare.
///
/// Belt and braces on purpose. The `Selection.h` change is the one that makes the error
/// impossible; this one makes the file structurally identical to the sibling that never hit it.
template<class Op,class T,int N,class Arch>
static int rank_of() { return sel::rank<Op,Key<T,N,Arch>>; }

template<class Op,int N,class Arch,int IS>
static int rank_of_mask() { return sel::rank<Op,Key<void,N,Arch,IS>>; }

template<class Op,class T,int N,class Arch,int IS>
static int rank_of_sel() { return sel::rank<Op,Key<T,N,Arch,IS>>; }

template<class Arch>
static void grid( const char *what ) {
    printf( "  %s: %s\n", what, Arch::name().c_str() );
    printf( "  %-16s%6s%6s%6s%6s%6s%6s%6s%6s%6s\n", "op \\ (type,lanes)",
            "f,4", "f,8", "f,16", "d,2", "d,4", "d,8", "i,4", "i,8", "i,16" );
    row<ops::fma,Arch>           ( "fma"           );
    row<ops::permute,Arch>       ( "permute"       );
    row<ops::bcast_lane<1>,Arch> ( "bcast_lane<1>" );
    row<ops::cmp_gt,Arch>        ( "cmp_gt"        );
    row<ops::cmp_lt,Arch>        ( "cmp_lt"        );
    row<ops::cmp_eq,Arch>        ( "cmp_eq"        );
    row<ops::cmp_ge,Arch>        ( "cmp_ge"        );
    printf( "\n" );
}

/// WHAT MUST NOT REGRESS. A template, not a plain function: `if constexpr` only discards the
/// untaken branch inside one, and in a plain function every assertion below would fire whatever
/// the target -- the trap `test_x86_dispatch.cpp` documents, and clang is the compiler that
/// catches it.
template<class Arch>
static void require_the_floor() {
    // ---- the ARMv7-A floor: what a 32-bit NEON part already has -----------------------------
    if constexpr ( Arch::template Has<features::NEON>::value ) {
        // ALL FOUR RELATIONS, SIGNED AND UNSIGNED, at a register width. This is the assertion
        // that would have caught a port written by mirroring x86: there, `cmp_ge` on an integer
        // and every unsigned comparison have no instruction and are synthesised. Here they are
        // `CMGE` and `CMHI`, so anything less than REGISTER means a row was left out.
        sel::require_at_least<ops::cmp_gt,Key<FP32,4,Arch>,sel::REGISTER>();
        sel::require_at_least<ops::cmp_lt,Key<FP32,4,Arch>,sel::REGISTER>();
        sel::require_at_least<ops::cmp_eq,Key<FP32,4,Arch>,sel::REGISTER>();
        sel::require_at_least<ops::cmp_ge,Key<FP32,4,Arch>,sel::REGISTER>();
        sel::require_at_least<ops::cmp_ge,Key<SI32,4,Arch>,sel::REGISTER>();
        sel::require_at_least<ops::cmp_gt,Key<PI32,4,Arch>,sel::REGISTER>();
        sel::require_at_least<ops::cmp_gt,Key<SI16,8,Arch>,sel::REGISTER>();
        sel::require_at_least<ops::cmp_gt,Key<PI8,16,Arch>,sel::REGISTER>();

        // `BSL` needs nothing beyond NEON, where x86's `blendv` waited for SSE4.1.
        sel::require_at_least<ops::select,Key<FP32,4,Arch,32>,sel::REGISTER>();
        sel::require_at_least<ops::select,Key<SI32,4,Arch,32>,sel::REGISTER>();
        sel::require_at_least<ops::select,Key<SI16,8,Arch,16>,sel::REGISTER>();

        // AN INTEGER MULTIPLY-ACCUMULATE. `MLA` is base NEON; x86 has no integer fma at any
        // feature level, so this cell is generic there by necessity and a register form here.
        sel::require_at_least<ops::fma,Key<SI32,4,Arch>,sel::REGISTER>();
        sel::require_at_least<ops::fma,Key<PI32,4,Arch>,sel::REGISTER>();

        sel::require_at_least<ops::mask_from_bits,Key<void,4,Arch>,sel::REGISTER>();
    }

    if constexpr ( Arch::template Has<features::NEON>::value && Arch::template Has<features::FMA>::value )
        sel::require_at_least<ops::fma,Key<FP32,4,Arch>,sel::REGISTER>();

    // ---- what only A64 adds -----------------------------------------------------------------
    if constexpr ( Arch::template Has<features::ASIMD>::value ) {
        // double precision is a LANE TYPE that ARMv7 does not have at all.
        sel::require_at_least<ops::cmp_gt,Key<FP64,2,Arch>,sel::REGISTER>();
        sel::require_at_least<ops::select,Key<FP64,2,Arch,64>,sel::REGISTER>();
        sel::require_at_least<ops::fma   ,Key<FP64,2,Arch>,sel::REGISTER>();
        // and the 64-bit integer compares.
        sel::require_at_least<ops::cmp_gt,Key<SI64,2,Arch>,sel::REGISTER>();
        sel::require_at_least<ops::cmp_gt,Key<PI64,2,Arch>,sel::REGISTER>();

        // `TBL` over a whole register, and `DUP` from any lane of one.
        sel::require_at_least<ops::permute   ,Key<FP32,4,Arch>,sel::REGISTER>();
        sel::require_at_least<ops::permute   ,Key<SI32,4,Arch>,sel::REGISTER>();
        // the NARROW types too, which is where `TBL` is the native operation rather than a
        // workaround -- and where the register impls had briefly made `permute` slower than it
        // was with no backend at all, because the generic form moves `values` one lane at a time.
        sel::require_at_least<ops::permute   ,Key<SI16,8,Arch>,sel::REGISTER>();
        sel::require_at_least<ops::permute   ,Key<PI16,8,Arch>,sel::REGISTER>();
        sel::require_at_least<ops::permute   ,Key<SI8,16,Arch>,sel::REGISTER>();
        sel::require_at_least<ops::permute   ,Key<PI8,16,Arch>,sel::REGISTER>();
        sel::require_at_least<ops::bcast_lane<1>,Key<FP32,4,Arch>,sel::REGISTER>();
        sel::require_at_least<ops::bcast_lane<3>,Key<SI32,4,Arch>,sel::REGISTER>();
        sel::require_at_least<ops::bcast_lane<1>,Key<FP64,2,Arch>,sel::REGISTER>();

        // `to_bits` at all four lane widths. There is no `movemask` on ARM, so each of these is
        // an AND plus an `ADDV` -- and the 16-lane one reduces its halves separately, because
        // `vaddvq_u8` returns a `uint8_t` and sixteen bits do not fit in one.
        sel::require_at_least<ops::to_bits,Key<void, 4,Arch,32>,sel::REGISTER>();
        sel::require_at_least<ops::to_bits,Key<void, 2,Arch,64>,sel::REGISTER>();
        sel::require_at_least<ops::to_bits,Key<void, 8,Arch,16>,sel::REGISTER>();
        sel::require_at_least<ops::to_bits,Key<void,16,Arch, 8>,sel::REGISTER>();
    }

    // ---- THE SPLIT FLOOR, which on ARM is the one that matters ------------------------------
    //
    // Every width above the register -- i.e. every width above four floats, i.e. the width the
    // README's own example asks for -- has to reach SPLIT. If any of these drops to GENERIC the
    // library is doing the one thing it exists not to do: going through the stack for a value
    // that fits in two registers.
    if constexpr ( Arch::template Has<features::ASIMD>::value ) {
        sel::require_at_least<ops::cmp_gt  ,Key<FP32, 8,Arch>,sel::SPLIT>();
        sel::require_at_least<ops::cmp_gt  ,Key<FP32,16,Arch>,sel::SPLIT>();
        sel::require_at_least<ops::cmp_gt  ,Key<FP64, 4,Arch>,sel::SPLIT>();
        sel::require_at_least<ops::cmp_ge  ,Key<SI32, 8,Arch>,sel::SPLIT>();
        sel::require_at_least<ops::select  ,Key<FP32, 8,Arch,32>,sel::SPLIT>();
        sel::require_at_least<ops::select  ,Key<FP32,16,Arch,32>,sel::SPLIT>();
        sel::require_at_least<ops::fma     ,Key<FP32, 8,Arch>,sel::SPLIT>();
        sel::require_at_least<ops::fma     ,Key<FP64, 4,Arch>,sel::SPLIT>();
        sel::require_at_least<ops::fma     ,Key<SI32, 8,Arch>,sel::SPLIT>();
        sel::require_at_least<ops::permute ,Key<FP32, 8,Arch>,sel::SPLIT>();
        sel::require_at_least<ops::permute ,Key<FP32,16,Arch>,sel::SPLIT>();
        sel::require_at_least<ops::to_bits ,Key<void, 8,Arch,32>,sel::SPLIT>();
        sel::require_at_least<ops::bcast_lane<1>,Key<FP32,8,Arch>,sel::SPLIT>();
        sel::require_at_least<ops::bcast_lane<5>,Key<FP32,8,Arch>,sel::SPLIT>();  // the UPPER half
        sel::require_at_least<ops::mask_from_bits,Key<void, 8,Arch>,sel::SPLIT>();
        sel::require_at_least<ops::mask_from_bits,Key<void,16,Arch>,sel::SPLIT>();
    }
}

int main() {
    printf( "  architecture   : %s\n", A::name().c_str() );
    printf( "  native widths  : float=%d double=%d SI32=%d SI16=%d SI8=%d\n\n",
            SimdSize<float,A>::value, SimdSize<double,A>::value, SimdSize<SI32,A>::value,
            SimdSize<SI16,A>::value, SimdSize<SI8,A>::value );

    grid<A> ( "this target   " );
    const int native_generic = nb_generic, native_cells = nb_cells;

    // THE SAME GRID AT THE ARMv7 FLOOR. Not decoration: it is what says the two feature levels
    // are really separate. If `ASIMD`-only instructions had been registered against `NEON`, this
    // column would look identical to the one above -- and a genuine 32-bit build would not
    // compile, which is exactly the failure the x86 lattice was fixed for.
    nb_generic = nb_cells = 0;
    grid<V7>( "ARMv7-A floor " );
    printf( "  (0 = GENERIC, a scalar loop; %d = SPLIT; %d = REGISTER; %d = MASK_REGISTER)\n\n",
            sel::SPLIT, sel::REGISTER, sel::MASK_REGISTER );

    // GUARDED ON WHETHER THE BACKEND EXISTS IN THIS TRANSLATION UNIT, not on whether the
    // architecture type claims the feature -- and the difference is the whole point of the
    // guard. `V7` is a TYPE: `ArmCpu<64,NEON,FMA>::Has<NEON>` is true on an x86 host too, while
    // `SimdOpsPlus_Neon.h` is `#if`'d out there for want of `<arm_neon.h>`, so every assertion
    // below would demand a register form that cannot exist and this file would not compile at
    // all off ARM.
    //
    // This is the same confusion, one architecture over, as `test_x86_dispatch.cpp` had when it
    // keyed its `fma` floor on `Has<FMA>` alone: a feature type says what the ARCHITECTURE has,
    // `ASIMD_ARM_HAS_NEON` says what this COMPILER may emit, and a floor needs both.
#ifdef ASIMD_ARM_HAS_NEON
    require_the_floor<A>();
    require_the_floor<V7>();
#endif

    // ---- the width you get when you do not name one -----------------------------------------
    //
    // `SimdVec<float>` is four lanes on every ARM part there has ever been, so unlike the x86
    // side there is no "widest hardware gets the slowest path" trap to fall into here. It is
    // asserted anyway, because that is the cell every caller who does not think about width lands
    // on.
    // EVERY RANK MATERIALISED FIRST, through the function templates above, and only then
    // compared -- see the note on `rank_of`. Nothing below names `sel::rank`.
    constexpr int nat = SimdSize<float,A>::value;
    const int n_gt   = rank_of     <ops::cmp_gt ,float,nat,A>();
    const int n_sel  = rank_of_sel <ops::select ,float,nat,A,32>();
    const int n_perm = rank_of     <ops::permute,float,nat,A>();
    const int n_bits = rank_of_mask<ops::to_bits,      nat,A,32>();
    const int n_fma  = rank_of     <ops::fma    ,float,nat,A>();
    printf( "  at the native width (%d lanes), SimdVec<float> gets:"
            " cmp_gt=%d select=%d permute=%d to_bits=%d fma=%d\n",
            nat, n_gt, n_sel, n_perm, n_bits, n_fma );

    if constexpr ( A::template Has<features::NEON>::value ) {
        CHECK( n_gt  > sel::GENERIC );
        CHECK( n_sel > sel::GENERIC );
    }
    if constexpr ( A::template Has<features::ASIMD>::value ) {
        CHECK( n_perm > sel::GENERIC );
        CHECK( n_bits > sel::GENERIC );
    }
    if constexpr ( A::template Has<features::NEON>::value && A::template Has<features::FMA>::value )
        CHECK( n_fma > sel::GENERIC );

    // ---- WHAT IS LEGITIMATELY GENERIC, and nothing else -------------------------------------
    //
    // Three cells, and they are all the same cell: `permute` on `double`, at 2, 4 and 8 lanes.
    //
    // The two-lane one is where it starts. `permute`'s index vector is always `SimdVec<SI32,N>`,
    // and at N = 2 that is 64 bits of a 128-bit register -- a width with no register impl of its
    // own -- so there is no `idx.data.reg` for a `TBL` control to be built from. x86 leaves the
    // identical cell generic, for the identical reason.
    //
    // The 4- and 8-lane ones follow from it rather than being separate holes: the split form
    // delegates to the halves, and `available` correctly refuses when the half has nothing better
    // than a lane loop to offer (§ 8.2 of FINDINGS.md -- a rank says "better if available", and
    // `available` has to mean it). Registering a `TBL`-based two-lane form would unlock all
    // three; whether it would actually beat the generic form at two lanes is unmeasured, and
    // guessing is what `available` exists to prevent.
    printf( "  %d of %d cells generic on this target, %d of %d at the ARMv7 floor.\n",
            native_generic, native_cells, nb_generic, nb_cells );
    if constexpr ( A::template Has<features::ASIMD>::value )
        CHECK( native_generic <= 3 );

    return report( "test_arm_dispatch" );
}
