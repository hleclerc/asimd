/* THE ARMv7-A FLOOR, AS A LIST OF INTRINSICS.
 *
 * Every intrinsic that `SimdVecImpl_Neon.h` and `SimdOpsPlus_Neon.h` register under
 * `features::NEON` ALONE -- i.e. everything the backend claims a 32-bit ARM part already has.
 * If one of these turns out to be A64-only, a genuine ARMv7 build fails to compile, on the one
 * machine nobody develops on. That is the exact failure the x86 feature lattice was rewritten
 * for ("half the SSE2 backend was really SSE4.1"), and this file is what stops it happening
 * again on the ARM side.
 *
 * DELIBERATELY C, AND COMPILED `-ffreestanding`. `arm_neon.h` is provided by the compiler, so
 * clang can compile this for `armv7a-linux-gnueabihf` with no cross toolchain -- but `arm_neon.h`
 * includes `<stdint.h>`, and clang's own `stdint.h` defers to the system one when
 * `__STDC_HOSTED__` is set. On an aarch64 Linux host that lands in the HOST's glibc, which then
 * fails on `bits/libc-header-start.h` because the armhf multiarch headers are not installed:
 *
 *     /usr/include/stdint.h:26:10: fatal error: 'bits/libc-header-start.h' file not found
 *
 * `-ffreestanding` clears `__STDC_HOSTED__`, so clang defines the integer types itself and glibc
 * never enters the picture. That is what makes this check run on any host rather than only on the
 * macOS one it was written on -- where clang's `stdint.h` is self-contained anyway, which is
 * exactly why the problem was invisible there.
 *
 * The pointer parameters below are spelled with the `<stdint.h>` typedefs rather than with
 * `long long` and friends, so they match the intrinsic signatures on both ABIs: `int64_t` is
 * `long` on LP64 and `long long` on ILP32.
 *
 * `arm_lattice_asimd.c` is the other half: the intrinsics that must NOT be available here.
 * `check_arm_lattice.sh` runs both and compares. */
#include <arm_neon.h>
/* dup, load, store, add, sub -- what ASIMD_NEON_COMMON registers for every type */
int32x4_t   t_s32( const int32_t *p, int32_t *q, int32x4_t a, int32x4_t b )   { vst1q_s32(q, vld1q_s32(p)); return vaddq_s32(vsubq_s32(a,b), vdupq_n_s32(3)); }
uint32x4_t  t_u32( const uint32_t *p, uint32_t *q, uint32x4_t a, uint32x4_t b ) { vst1q_u32(q, vld1q_u32(p)); return vaddq_u32(vsubq_u32(a,b), vdupq_n_u32(3)); }
float32x4_t t_f32( const float *p, float *q, float32x4_t a, float32x4_t b ) { vst1q_f32(q, vld1q_f32(p)); return vaddq_f32(vsubq_f32(a,b), vdupq_n_f32(3)); }
int64x2_t   t_s64( const int64_t *p, int64_t *q, int64x2_t a, int64x2_t b ) { vst1q_s64(q, vld1q_s64(p)); return vaddq_s64(vsubq_s64(a,b), vdupq_n_s64(3)); }
uint64x2_t  t_u64( const uint64_t *p, uint64_t *q, uint64x2_t a, uint64x2_t b ) { vst1q_u64(q, vld1q_u64(p)); return vaddq_u64(vsubq_u64(a,b), vdupq_n_u64(3)); }
int16x8_t   t_s16( const int16_t *p, int16_t *q, int16x8_t a, int16x8_t b ) { vst1q_s16(q, vld1q_s16(p)); return vaddq_s16(vsubq_s16(a,b), vdupq_n_s16(3)); }
uint16x8_t  t_u16( const uint16_t *p, uint16_t *q, uint16x8_t a, uint16x8_t b ) { vst1q_u16(q, vld1q_u16(p)); return vaddq_u16(vsubq_u16(a,b), vdupq_n_u16(3)); }
int8x16_t   t_s8 ( const int8_t *p, int8_t *q, int8x16_t a, int8x16_t b ) { vst1q_s8(q, vld1q_s8(p)); return vaddq_s8(vsubq_s8(a,b), vdupq_n_s8(3)); }
uint8x16_t  t_u8 ( const uint8_t *p, uint8_t *q, uint8x16_t a, uint8x16_t b ) { vst1q_u8(q, vld1q_u8(p)); return vaddq_u8(vsubq_u8(a,b), vdupq_n_u8(3)); }
/* mul, min/max, and, variable shl, compares, bsl, mla */
float32x4_t m_f32( float32x4_t a, float32x4_t b ) { return vminq_f32(vmaxq_f32(vmulq_f32(a,b),b),a); }
int32x4_t   m_s32( int32x4_t a, int32x4_t b )   { return vminq_s32(vmaxq_s32(vmulq_s32(a,b),b),vandq_s32(a,b)); }
uint32x4_t  m_u32( uint32x4_t a, uint32x4_t b ) { return vminq_u32(vmaxq_u32(vmulq_u32(a,b),b),vandq_u32(a,b)); }
int16x8_t   m_s16( int16x8_t a, int16x8_t b )   { return vminq_s16(vmaxq_s16(vmulq_s16(a,b),b),vandq_s16(a,b)); }
uint16x8_t  m_u16( uint16x8_t a, uint16x8_t b ) { return vminq_u16(vmaxq_u16(vmulq_u16(a,b),b),vandq_u16(a,b)); }
int8x16_t   m_s8 ( int8x16_t a, int8x16_t b )   { return vminq_s8(vmaxq_s8(vmulq_s8(a,b),b),vandq_s8(a,b)); }
uint8x16_t  m_u8 ( uint8x16_t a, uint8x16_t b ) { return vminq_u8(vmaxq_u8(vmulq_u8(a,b),b),vandq_u8(a,b)); }
int64x2_t   a_s64( int64x2_t a, int64x2_t b )   { return vandq_s64(a,b); }
uint64x2_t  a_u64( uint64x2_t a, uint64x2_t b ) { return vandq_u64(a,b); }
int32x4_t   s_s32( int32x4_t a, int32x4_t b )   { return vshlq_s32(a,b); }
int64x2_t   s_s64( int64x2_t a, int64x2_t b )   { return vshlq_s64(a,b); }
int16x8_t   s_s16( int16x8_t a, int16x8_t b )   { return vshlq_s16(a,b); }
int8x16_t   s_s8 ( int8x16_t a, int8x16_t b )   { return vshlq_s8(a,b); }
uint32x4_t  s_u32( uint32x4_t a, uint32x4_t b ) { return vshlq_u32(a, vreinterpretq_s32_u32(b)); }
uint64x2_t  s_u64( uint64x2_t a, uint64x2_t b ) { return vshlq_u64(a, vreinterpretq_s64_u64(b)); }
uint16x8_t  s_u16( uint16x8_t a, uint16x8_t b ) { return vshlq_u16(a, vreinterpretq_s16_u16(b)); }
uint8x16_t  s_u8 ( uint8x16_t a, uint8x16_t b ) { return vshlq_u8 (a, vreinterpretq_s8_u8 (b)); }
float32x4_t f_f32( float32x4_t a, float32x4_t b ) { return vreinterpretq_f32_u32( vandq_u32( vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b) ) ); }
uint32x4_t  c_f32( float32x4_t a, float32x4_t b ) { return vandq_u32(vandq_u32(vcgtq_f32(a,b),vcltq_f32(a,b)),vandq_u32(vceqq_f32(a,b),vcgeq_f32(a,b))); }
uint32x4_t  c_s32( int32x4_t a, int32x4_t b )   { return vandq_u32(vandq_u32(vcgtq_s32(a,b),vcltq_s32(a,b)),vandq_u32(vceqq_s32(a,b),vcgeq_s32(a,b))); }
uint32x4_t  c_u32( uint32x4_t a, uint32x4_t b ) { return vandq_u32(vandq_u32(vcgtq_u32(a,b),vcltq_u32(a,b)),vandq_u32(vceqq_u32(a,b),vcgeq_u32(a,b))); }
uint16x8_t  c_s16( int16x8_t a, int16x8_t b )   { return vandq_u16(vandq_u16(vcgtq_s16(a,b),vcltq_s16(a,b)),vandq_u16(vceqq_s16(a,b),vcgeq_s16(a,b))); }
uint8x16_t  c_u8 ( uint8x16_t a, uint8x16_t b ) { return vandq_u8 (vandq_u8 (vcgtq_u8 (a,b),vcltq_u8 (a,b)),vandq_u8 (vceqq_u8 (a,b),vcgeq_u8 (a,b))); }
float32x4_t b_f32( uint32x4_t m, float32x4_t a, float32x4_t b ) { return vbslq_f32(m,a,b); }
int64x2_t   b_s64( uint64x2_t m, int64x2_t a, int64x2_t b )     { return vbslq_s64(m,a,b); }
uint8x16_t  b_u8 ( uint8x16_t m, uint8x16_t a, uint8x16_t b )   { return vbslq_u8(m,a,b); }
int32x4_t   l_s32( int32x4_t c, int32x4_t a, int32x4_t b ) { return vmlaq_s32(c,a,b); }
int8x16_t   l_s8 ( int8x16_t c, int8x16_t a, int8x16_t b ) { return vmlaq_s8(c,a,b); }
uint32x4_t  k_u32( uint32_t b ) { const uint32_t s[4]={1,2,4,8}; uint32x4_t sel=vld1q_u32(s);
    return vceqq_u32( vandq_u32( vdupq_n_u32(b), sel ), sel ); }
