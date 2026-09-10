#pragma once

// =====================================================================================
// VARIANT SELECTION.
//
// An operation usually has several possible implementations for a given (type, width,
// architecture): a generic one that walks lanes, one that splits across several registers, one
// that maps to a dedicated instruction, sometimes a better one still on targets with mask
// registers. Something has to pick.
//
// = WHY NOT PLAIN OVERLOADING WITH `requires`
//
//   CONSTRAINTS DO NOT ORDER EACH OTHER. Adding an AVX-512 form next to an AVX one makes the
//   call AMBIGUOUS as soon as both features are present. Writing the constraint as a conjunction
//   does not help: two `requires` clauses WRITTEN SEPARATELY yield distinct atomic constraints
//   even when textually identical, so there is no subsumption. Working around it means rebuilding
//   an ordering by hand, one operation at a time.
//
//   OVERLOAD RESOLUTION DEPENDS ON INCLUDE ORDER. A facade calling `internal::f(...)` through a
//   QUALIFIED name freezes resolution at its own definition: a register form declared later is
//   invisible. The code still compiles, still returns correct results, and silently loses all
//   vectorization. This happened while porting a polygon clipping kernel -- it ran at scalar
//   speed with not one `vpermps` in the binary.
//
//   AVAILABLE IS NOT BEST. On a Skylake-X, 512-bit exists and downclocks: it is slower than
//   256-bit. An instruction-set hierarchy therefore cannot serve as a preference order --
//   preference is a MEASUREMENT, not a generation.
//
// = THE THREE CHOICES HERE
//
//   AN EXPLICIT RANK, hence a TOTAL order: never ambiguous, and a preference can be corrected
//   when a benchmark disproves it, without touching any constraint.
//
//   A VARIANT IS A CLASS SPECIALIZATION, never a function overload: specializations are looked up
//   at the point of INSTANTIATION rather than of definition, so include order stops deciding.
//
//   THE CHOICE IS ASSERTABLE. This is the important one, and it comes from a scar: vectorization
//   was silently lost TWICE in that port -- once through include order, once because a union sent
//   every vector back to memory. A mechanism that picks without letting you check what it picked
//   only moves the problem. `require_at_least<...>` turns a silent fallback into a compile error.
//
// = WHEN TO USE A RANK, AND WHEN NOT TO
//
//   Use one when several implementations are viable FOR THE SAME ARGUMENT TYPES -- that is where
//   ambiguity lives. When the argument types already tell the implementations apart (a bit mask
//   versus a lane mask, say), ordinary overloading is unambiguous and a rank would only add noise.
//
// = KNOWN LIMITS
//
//   A specialization declared after an actual INSTANTIATION is ill-formed, no diagnostic required.
//   GCC does not report it and does what you would expect, but that is not guaranteed. So the
//   discipline "declare every backend before any use" still holds -- a single `backends.h`
//   included at a fixed point is enough -- and `require_at_least` is the safety net.
//
//   THE ORDER IS TOTAL BETWEEN RANKS, NOT WITHIN ONE. Two backends registering the same
//   (Op, Key, RANK) are as ambiguous as two overloads would be -- the rank buys nothing there.
//   Met in practice: a `permute` on four floats exists as `pshufb` (SSSE3, four instructions) and
//   as `vpermilps` (AVX, one), and a machine with AVX has both. Two ways out, and the choice
//   says something:
//     - give them different ranks, if one is simply preferable -- but REGISTER is one level, and
//       inventing 19 and 21 turns a readable scale into a pecking order;
//     - make the constraints MUTUALLY EXCLUSIVE, `Has<SSSE3> && ! Has<AVX>`, which states the
//       real relationship: the older form is the fallback, not a rival.
//   The second is what this codebase does. `ASIMD_X86_REQ_EXCL` in `SimdOpsPlus_X86.h` spells it.
// =====================================================================================

namespace asimd {
namespace sel {

/// Ranks. These are not instruction-set generations but PREFERENCES. A backend may declare itself
/// unavailable for one specific micro-architecture (512-bit on Skylake-X) without anything else
/// moving, since the key carries the `Arch`.
enum : int {
    GENERIC       =  0,   ///< lane by lane. Always available: this is the guaranteed fallback.
    SPLIT         = 10,   ///< several registers for one requested width -- what asimd is for
    REGISTER      = 20,   ///< a dedicated instruction
    MASK_REGISTER = 30,   ///< better still on targets with mask registers
    MAX_RANK      = 40
};

/// A variant. A backend registers by specializing this, with `available = true` and a `run`.
template<class Op, class Key, int RANK>
struct Variant { static constexpr bool available = false; };

template<class Op, class Key, int R>
constexpr int search() {
    if constexpr ( R < 0 )                            return -1;
    else if constexpr ( Variant<Op,Key,R>::available ) return R;
    else                                              return search<Op,Key,R-1>();
}

/// The selected rank -- a constant, hence printable, comparable, assertable.
template<class Op, class Key>
inline constexpr int rank = search<Op,Key,MAX_RANK>();

template<class Op, class Key, class... A>
decltype( auto ) call( A &&...a ) { return Variant<Op,Key,rank<Op,Key>>::run( static_cast<A&&>( a )... ); }

/// THE SAFETY NET. "On this target I want at least that much": an unexpected fallback becomes a
/// compile error instead of a 30 % loss you find out about on a benchmark, or never.
template<class Op, class Key, int MIN>
constexpr void require_at_least() {
    static_assert( rank<Op,Key> >= MIN, "no variant good enough for this target" );
}

} // namespace sel
} // namespace asimd
