#pragma once

#include "SimdFeatureOn.h"

namespace asimd {

namespace features {

#define ASIMD_CMON_TYPES float,double,std::int8_t,std::int16_t,std::int32_t,std::int64_t,std::uint8_t,std::uint16_t,std::uint32_t,std::uint64_t

struct AVX512 : SimdFeatureOn<512,32,ASIMD_CMON_TYPES> {
    static std::string name() { return "AVX512"; }
};

struct AVX2 : SimdFeatureOn<256,16,ASIMD_CMON_TYPES> {
    static std::string name() { return "AVX2"; }
};

struct AVX : SimdFeatureOn<256,16,ASIMD_CMON_TYPES> {
    static std::string name() { return "AVX"; }
};

struct SSE2 : SimdFeatureOn<128,8,ASIMD_CMON_TYPES> {
    static std::string name() { return "SSE2"; }
};

struct SSE : SimdFeatureOn<128,8,ASIMD_CMON_TYPES> {
    static std::string name() { return "SSE"; }
};

#undef ASIMD_CMON_TYPES

} // namespace features

} // namespace asimd
