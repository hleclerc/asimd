#pragma once

#include "GenericFeatures.h"
#include "SimdFeatureOn.h"
#include "FeatureSet.h"

namespace asimd {

namespace features {

#define ASIMD_CMON_TYPES float,double,std::int8_t,std::int16_t,std::int32_t,std::int64_t,std::uint8_t,std::uint16_t,std::uint32_t,std::uint64_t

/// 128 bits, 32 registers on AArch64. Declared here so `NativeCpu` can name the width before any
/// backend exists: `SimdVec<float>` is then four lanes running through the generic forms, which
/// is correct and slow, rather than a compile error.
struct NEON : SimdFeatureOn<128,32,ASIMD_CMON_TYPES> { static std::string name() { return "NEON"; } };

#undef ASIMD_CMON_TYPES

} // namespace features

/**
*/
template<int ptr_size_in_bits,class... Features>
struct ArmCpu : FeatureSet<Features...> {
    using                 size_type = typename std::conditional<ptr_size_in_bits==64,std::uint64_t,std::uint32_t>::type;
    static constexpr bool cpu       = true;

    static std::string    name      () { return "Arm<" + std::to_string( ptr_size_in_bits ) + FeatureSet<Features...>::feature_names() + ">"; }
};

} // namespace asimd
