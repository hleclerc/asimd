/* THE OTHER HALF OF THE LATTICE: the intrinsics `SimdVecImpl_Neon.h` and `ops/Neon.h`
 * register under `features::ASIMD`, i.e. the ones that are A64-only.
 *
 * THIS FILE IS EXPECTED TO FAIL on an ARMv7 target, and that is the assertion. A negative check
 * is worth as much as the positive one: if these turned out to be available on ARMv7 after all,
 * the `ASIMD` guard would be costing a 32-bit part instructions it actually has -- and if a
 * FUTURE edit moved one of them under `NEON`, `arm_lattice_neon.c` would start failing there and
 * say so. Between the two files the split is pinned from both sides.
 *
 * `check_arm_lattice.sh` compiles this for both targets and expects: AArch64 yes, ARMv7 no. */
#include <arm_neon.h>

/* double-precision LANES -- ARMv7 NEON has none at all */
float64x2_t d_f64( const double *p, double *q, float64x2_t a, float64x2_t b ) {
    vst1q_f64( q, vld1q_f64( p ) );
    return vminq_f64( vmaxq_f64( vmulq_f64( vaddq_f64( a, b ), b ), a ), vdupq_n_f64( 1 ) );
}
/* floating point DIVIDE */
float32x4_t d_f32( float32x4_t a, float32x4_t b ) { return vdivq_f32( a, b ); }
/* the horizontal reductions: ADDV, UMAXV, UMINV */
float   h_f32( float32x4_t a )  { return vaddvq_f32( a ); }
int     h_s32( int32x4_t a )    { return vaddvq_s32( a ); }
unsigned r_u32( uint32x4_t a )  { return vmaxvq_u32( a ) + vminvq_u32( a ); }
/* TBL over a whole 128-bit register */
uint8x16_t t_u8( uint8x16_t v, uint8x16_t c ) { return vqtbl1q_u8( v, c ); }
/* DUP from an arbitrary lane of a q register */
float32x4_t l_f32( float32x4_t a ) { return vdupq_laneq_f32( a, 3 ); }
/* the 64-bit integer comparisons */
uint64x2_t c_s64( int64x2_t a, int64x2_t b ) { return vcgtq_s64( a, b ); }
