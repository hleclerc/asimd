// THE SELECTION MECHANISM (`Selection.h`), and the three properties that justify it.
//
// Two halves: the mechanism itself, exercised with dummy variants so the properties are pinned
// down independently of any backend; then the ranks actually selected for the real operations on
// the machine this is compiled for.
#include <asimd/SimdOpsPlus.h>

#include "check.h"

namespace sel = asimd::sel;

// ============ part one: the mechanism ============
struct Permute {};
template<class T,int N,class Arch> struct Key {};
struct Avx512 {}; struct Avx2 {}; struct Neon {}; struct Sve {};

// rank 0: the fallback, ALWAYS available. This is what guarantees there is always an answer.
template<class T,int N,class Arch>
struct sel::Variant<Permute,Key<T,N,Arch>,sel::GENERIC> {
    static constexpr bool available = true;
    static const char *run() { return "generic"; }
};

// rank 20: a dedicated instruction
template<class T> struct sel::Variant<Permute,Key<T,8,Avx2>,sel::REGISTER> {
    static constexpr bool available = true;
    static const char *run() { return "vpermps"; }
};
template<class T> struct sel::Variant<Permute,Key<T,8,Avx512>,sel::REGISTER> {
    static constexpr bool available = true;
    static const char *run() { return "vpermps"; }
};

// rank 30: better still on that target. NO AMBIGUITY with rank 20 -- the order is TOTAL, where
// two separately written `requires` clauses would not order each other at all.
template<class T> struct sel::Variant<Permute,Key<T,8,Avx512>,sel::MASK_REGISTER> {
    static constexpr bool available = true;
    static const char *run() { return "vpermps + mask registers"; }
};

// ---- A FACADE, DEFINED HERE, before the SVE variant below. This is exactly the trap that cost
// all vectorization with function overloads and a qualified call.
template<class Arch>
const char *permute_on() { return sel::call<Permute,Key<float,8,Arch>>(); }

template<class T> struct sel::Variant<Permute,Key<T,8,Sve>,sel::MASK_REGISTER> {
    static constexpr bool available = true;
    static const char *run() { return "vrgather + predicates"; }
};

static bool same( const char *a, const char *b ) {
    while ( *a && *a == *b ) { ++a; ++b; }
    return *a == *b;
}

int main() {
    // ranks lifted out of the macro bodies: a comma inside a template argument does not survive
    // the preprocessor.
    constexpr int r_avx512 = sel::rank<Permute,Key<float,8,Avx512>>;
    constexpr int r_avx2   = sel::rank<Permute,Key<float,8,Avx2>>;
    constexpr int r_neon   = sel::rank<Permute,Key<float,8,Neon>>;
    constexpr int r_sve    = sel::rank<Permute,Key<float,8,Sve>>;

    // 1. TOTAL ORDER: the best-ranked one wins, with no possible ambiguity.
    CHECK( r_avx512 == sel::MASK_REGISTER );
    CHECK( same( permute_on<Avx512>(), "vpermps + mask registers" ) );
    CHECK( r_avx2 == sel::REGISTER );
    CHECK( same( permute_on<Avx2>(), "vpermps" ) );

    // 2. GUARANTEED FALLBACK: a target with no register form lands on the generic one.
    CHECK( r_neon == sel::GENERIC );
    CHECK( same( permute_on<Neon>(), "generic" ) );

    // 3. INDEPENDENCE FROM DECLARATION ORDER: the SVE variant is declared AFTER the facade that
    // uses it, and is still selected. A class specialization is looked up at the point of
    // INSTANTIATION, not of definition.
    CHECK( r_sve == sel::MASK_REGISTER );
    CHECK( same( permute_on<Sve>(), "vrgather + predicates" ) );

    // 4. THE SAFETY NET. It passes here; uncommenting the second line must FAIL THE BUILD -- that
    // is the whole point: an unexpected fallback stops being silent.
    sel::require_at_least<Permute,Key<float,8,Avx512>,sel::REGISTER>();
    // sel::require_at_least<Permute,Key<float,8,Neon>,sel::REGISTER>();   // <- must not compile

    printf( "  dummy variants : AVX512=%d AVX2=%d NEON=%d SVE=%d\n",
            r_avx512, r_avx2, r_neon, r_sve );

    // ============ part two: the real operations, on this machine ============
    //
    // CONDITIONAL ON THE FEATURES, and it has to be. These used to assert unconditionally that
    // `float x 8` had a register form -- true on the AVX2 box this was written on, false under
    // -msse2 where eight floats are two registers and `vpermps` does not exist. The assertion
    // was not wrong about the goal, only about which target it applied to; `tests/run_all_isa.sh`
    // is what made the difference visible.
    {
        using namespace asimd;
        using A = NativeCpu;
        constexpr int r_perm = sel::rank<ops::permute,asimd::Key<float,8,A>>;
        constexpr int r_gt   = sel::rank<ops::cmp_gt,asimd::Key<float,8,A>>;
        constexpr int r_fma  = sel::rank<ops::fma,asimd::Key<float,8,A>>;
        printf( "  real ops here  : permute=%d cmp_gt=%d fma=%d\n", r_perm, r_gt, r_fma );

        // Nothing may silently drop to the generic form on a machine that HAS the instructions.
        if constexpr ( A::template Has<features::AVX2>::value ) CHECK( r_perm > sel::GENERIC );
        if constexpr ( A::template Has<features::AVX >::value ) CHECK( r_gt   > sel::GENERIC );
        if constexpr ( A::template Has<features::AVX2>::value
                    && A::template Has<features::FMA >::value ) CHECK( r_fma  > sel::GENERIC );
    }

    return report( "test_selection" );
}
