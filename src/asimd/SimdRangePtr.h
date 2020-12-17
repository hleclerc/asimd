#pragma once

#include "internal/N.h"
#include "SimdAlig.h"
#include "SimdVec.h"
#include "Ptr.h"

namespace asimd {

/**
  Ptrs must contain
  * static constexpt ... alignment => minimum _known_ alignment, in bits
  * T *get() => the pointer
*/
template<class Arch,bool runtime_alig_test=true>
struct SimdRangePtr {
    /// func( TI index, N<simd_size>, N<ptr_0_is_aligned>, N<ptr_1_is_aligned>... )
    /// Version with beg % size assumed to be 0
    template<class TI,class Func,class... Ptrs>
    static void for_each_with_beg_aligned( TI beg, TI end, Func &&func, Ptrs... /*ptrs*/ ) {
        _fewba( beg, end, std::forward<Func>( func ), N<( Ptrs::alignment >= SimdAlig<typename Ptrs::T,Arch>::value )>()... );
    }

private:
    template<class TI,class Func,class... Als>
    static void _fewba( TI beg, TI end, Func &&func, Als... als ) {
        func( beg, N<0>(), als... );
        //        for( TI cur = beg, nxt; ; cur = nxt ) {
        //            nxt = cur + size;
        //            if ( nxt > end )
        //                return SimdRange<next_size>::for_each_with_beg_aligned( cur, end, std::forward<Func>( func ) );

        //            func( cur, N<size>() );
        //        }
    }
};

} // namespace asimd
