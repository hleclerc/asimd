# `asimd` — the width is the author's choice

A portable SIMD layer where **the vector width is a parameter of the type**, not a consequence of
the target:

```cpp
using V = asimd::SimdVec<float,8>;      // eight lanes. Everywhere.
```

On AVX2 that is one `ymm`. On SSE4 it is **two `xmm`** — `SimdVecImpl` splits recursively
(`split_size_0 = prev_pow_2(size)`, the rest into `v1`, down to a single lane). Neither xsimd nor
Highway does this: Highway refuses at compile time (*"Too many lanes"*) as soon as the descriptor
exceeds the target, which rules out NEON — 128 bits, so four `float` lanes — for any kernel
written on eight.

On **NEON that line above is two `q` registers**, and it is not a special case: Advanced SIMD has
been 128 bits since 2005 and still is, so on ARM every width past four floats is a split. The
premise the library is built on is the ordinary situation on that architecture rather than the
exception — measured at 1.07 ns/op for an eight-lane `permute` against 1.89 for the generic
fallback, and 0.59 against 0.85 for `to_bits`.

Header-only, C++20.

```cpp
#include <asimd/SimdOpsPlus.h>

using V = asimd::SimdVec<float,8>;
using I = asimd::SimdVec<asimd::SI32,8>;

V s = asimd::fma( V( dx ), vx, asimd::fma( V( dy ), vy, V( -off ) ) );
unsigned outside = asimd::to_bits( s > V( 0.f ) );          // the sign mask, as an integer
V t = asimd::permute( v, I::load_aligned( idx ) );          // variable-index permutation
V u = asimd::select( asimd::mask_from_bits<8>( 1u << k ), a, b );
```

---

## 1. Operations

| | |
|---|---|
| `fma(a,b,c)` | fused multiply-add |
| `to_bits(m)` | a mask **as an integer** — this is what carries `ctz`, `popcount` and rotations |
| `mask_from_bits<n>(b)` | the dual: one `kmovb` where `eq(iota,i)` costs two instructions |
| `select(m,a,b)` | blend two vectors according to a mask |
| `permute(v,idx)` | **variable-index** permutation |
| `bcast_lane<i>(v)` | broadcast a compile-time lane |
| `gt` `lt` `eq` `ge` | comparisons, materialized as a mask |
| `-` `*` `/` `&` | the operators |
| `<<` | a **per-lane** shift, each lane by its own amount — one instruction on ARM at every width, and none on x86 below AVX2 |

Comparisons are **lazy**: `a > b` computes nothing, it returns an object holding both operands,
and `to_bits` or `select` decides in which form to materialize it. That is finer than what
intrinsics and the other libraries do, where a comparison picks its representation before knowing
what it will be used for.

## 2. Variant selection

Every operation goes through `Selection.h`. A backend registers by specializing
`sel::Variant<Op, Key, RANK>`; the selector takes the highest-ranked available one, and rank 0 —
the generic form — is always available, so there is always an answer.

```cpp
sel::rank<ops::permute, Key<float,8,NativeCpu>>          // 20 (REGISTER) on AVX2
sel::require_at_least<ops::cmp_gt, Key<float,8,A>, sel::MASK_REGISTER>();
```

Three problems this solves, all met in practice:

**Constraints do not order each other.** Adding an AVX-512 form next to an AVX one makes the call
ambiguous as soon as both features are present. Writing the constraint as a conjunction does not
help: two `requires` clauses written separately yield *distinct* atomic constraints even when
textually identical, so there is no subsumption. An explicit rank gives a total order instead.

**Overload resolution depends on include order.** A facade calling `internal::f(...)` through a
qualified name freezes resolution at its own definition; a register form declared later is
invisible. The code still compiles, still returns correct results, and silently loses all
vectorization. Class specializations are looked up at instantiation, so order stops mattering.

**Available is not best.** On a Skylake-X, 512-bit exists and downclocks. Preference is a
measurement, not an instruction-set generation — so it is declared, and a backend can rule itself
out for one micro-architecture without anything else moving.

And `require_at_least` turns a silent fallback into a compile error, which matters more than it
sounds: vectorization was lost twice, silently, while this was being built.

## 3. The layout of a register impl decides the ABI

This one was a performance bug, not a correctness bug. A plain `fma` passed by value cost **11
instructions including 3 memory accesses**, against 4 for raw intrinsics. Every vector went through
the stack on every call.

The SysV x86-64 rule: an aggregate larger than two *eightbytes* is passed in registers only if the
first eightbyte is `SSE` and **all** the following ones are `SSEUP`. But `float[8]` classifies as
`SSE,SSE,SSE,SSE` — an array is not a vector — and a union takes the **worst** class among its
members.

| the union holds | instructions | memory accesses |
|---|---|---|
| the register alone | 4 | 0 |
| **a vector type + the register** | **4** | **0** |
| `T values[SIZE]` + the register | 8 | 4 |
| … + `Split` (two `__m128`) | 8 | 4 |

`Split` is unrecoverable: two `__m128` give `SSE,SSEUP,SSE,SSEUP`, and the third eightbyte alone
sends everything back to memory.

So in `SIMD_VEC_IMPL_REG`: `values` typed with `vector_size` (which keeps `values[i]` indexing) and
**`Split` taken out of the union**. Operations without a register form now go lane by lane through
`values`. `iota` had to follow, and got register forms — without them gcc materialized it as six
consecutive `vpinsrd` rather than a constant load.

Two things the audit added to this section. First, the same rule applies to **masks**, and had not
been applied: `SIMD_MASK_IMPL_REG_LARGE` still held `PI32 values[8]` *and* a `Split`, so every
lane-flavoured mask crossing a call went through memory — 9 instructions and 3 stack accesses on a
by-value `select`, against 4 and 0 once fixed. The rule is now in one place,
`support/VecValues.h`, which both macros use and which carries the MSVC fallback (MSVC has no
`vector_size`, and does not need one: its x64 convention passes vectors by reference anyway, so
the classification question never arises).

Second, taking `Split` out of the union had a consequence nobody followed through: the generic
fallbacks still recursed into `data.split`, which a register impl no longer has. At a
register-backed width, any operation *without* a register form was therefore a hard compile error
rather than a slow path — `.sum()`, integer `mul` and `div`, the strided `iota`, `gather` below
AVX2. A `HasSplit` concept and an `if constexpr` fix it, and buy something on the way: when there
is no split but `values` is a vector type, the operation applies **to the whole vector**, and gcc
and clang lower it themselves. That is how `SI32 × 8` gets a `vpmulld` with no backend registered
for it.

## 4. ARM

Two backends now, and the second one is not a transcription of the first. `architectures/ArmCpu.h`
and `ArmCpuFeatures.h` declare the target, `impl/SimdVecImpl_Neon.h` and `SimdMaskImpl_Neon.h`
carry the vectors and the masks, `SimdOpsPlus_Neon.h` the operations of § 1. The variant *shapes*
moved to `SimdOpsPlus_Shapes.h`, shared with x86: what a variant of `select` looks like is a
property of `Selection.h`, not of an instruction set, so a backend is now a **table** and nothing
else.

### The lattice splits at A64, not at "NEON"

`__ARM_NEON` is defined on ARMv7-A **and** on AArch64, and about a third of what a 128-bit backend
wants to call does not exist on the first one:

| | |
|---|---|
| `float64x2_t` | no double-precision **lanes** at all on ARMv7 |
| `vdivq_f32` | no vector floating point divide |
| `vaddvq_*`, `vmaxvq_*` | the horizontal reductions are A64 |
| `vqtbl1q_u8` | `TBL` over a whole register is A64; ARMv7's reads a 64-bit table |
| `vcgtq_s64` | 64-bit integer compares are A64 |
| 16 vs 32 | q-registers |

Writing all of that under one feature would repeat exactly the mistake the x86 lattice was fixed
for — guarded by `__SSE2__`, half the SSE2 backend was really SSE4.1, and a genuine SSE2 target
did not build. So there are **two** width-carrying features, `NEON` and `ASIMD`, and the split is
where the instruction set splits. `tests/check_arm_lattice.sh` checks it **from both sides**:

```
clang: does the ARM feature lattice hold?
  NEON-guarded intrinsics on ARMv7-A             yes  ok
  NEON-guarded intrinsics on AArch64             yes  ok
  ASIMD-guarded intrinsics on AArch64            yes  ok
  ASIMD-guarded intrinsics on ARMv7-A            no   ok
```

The last row is the one that earns its place: if those intrinsics *were* available on ARMv7, the
`ASIMD` guard would be denying a 32-bit part instructions it has. Both probe files are C, and
`arm_neon.h` comes from the compiler, so this needs **no cross toolchain** — it runs on a laptop
rather than only in CI.

`-ffreestanding` is what makes that true off macOS, and it took a red CI run to find out.
`arm_neon.h` includes `<stdint.h>`; clang's own `stdint.h` defers to the system one whenever
`__STDC_HOSTED__` is set, and on an aarch64 Linux host that lands in the **host's** glibc and dies
on `bits/libc-header-start.h`, because the armhf multiarch headers are not installed. Freestanding
clears `__STDC_HOSTED__`, clang defines the integer types itself, and the only headers left in the
picture are its own — verified with `-H`. On macOS clang's `stdint.h` is self-contained anyway,
which is exactly why the problem was invisible where the check was written.

And when a host still cannot compile for the target, the script now says **`SKIPPED`** and names
the file it could not find. The first version reported `EXPECTED yes` and
`THE LATTICE IS WRONG` — sending the reader to look for a bug in `ArmCpuFeatures.h` that was not
there. A missing intrinsic and a missing header are told apart by whether a *file* was not found,
not by `fatal error:` alone: the ASIMD-on-ARMv7 row is *meant* to fail, it fails with dozens of
errors, and clang caps that with `fatal error: too many errors emitted` — which the first attempt
at the discriminator duly misread. CI passes `--strict`, where a skipped row is a failure too,
because a green tick nobody reads is how that row would quietly stop being checked.

The extensions above Advanced SIMD — `FP16`, `BF16`, `DOTPROD`, `I8MM`, `RDM`, `FCMA`, `SVE`,
`SVE2` — are declared as markers and dispatched on by nothing yet, with the reason written next to
each. Two are worth naming here. **`DOTPROD` and `I8MM` reduce**: `SDOT` accumulates four 8-bit
products into each 32-bit lane, which is not a variant of any operation `SimdVec` has, but a new
one. And **`SVE` deliberately carries no width**: an SVE vector is 128 to 2048 bits and the length
is not known until the program runs, so `svfloat32_t` is a *sizeless* type — it cannot be a class
member, hence cannot sit in the union `SIMD_VEC_IMPL_REG` builds. Giving it a width here would
make `SimdSize<float>` a number the hardware may contradict. `-msve-vector-bits=N` is the door,
and it opens onto a backend, not a declaration.

### What is better here than on x86

A port written by mirroring would have missed all five:

| | |
|---|---|
| `vcgtq_u32` &c. | the **unsigned** compares are instructions. x86 has none below AVX-512: it flips the sign bit of both operands and uses the signed one — four instructions where this is one |
| `vcgeq_*` | all four relations exist for every type. On x86, `cmp_ge` on an integer is `not( a < b )` |
| `vmlaq_s32` | an **integer** multiply-accumulate, one instruction. x86 has no integer fma at any feature level — `ops::fma` on `SI32` is one of the four cells this README used to list as honestly generic |
| `vshlq_*` | a **per-lane variable shift** at every width. x86 has none below AVX2, and none at all on the 8- and 16-bit types |
| `ADDV` / `UMAXV` | a horizontal reduction in **one** instruction, where SSE2 needs a shuffle ladder |

And one thing that is worse: **there is no `movemask`, and no mask register either.** A comparison
always yields a lane mask, so `to_bits` is a *reduction* — AND with `{1,2,4,8}`, then `ADDV`. Two
instructions and a constant against one `movmskps`. This README predicted "three or four"; two is
as good as it gets, and it is the only operation where ARM needs more instructions than x86 for
the same answer.

### Measured

Instructions for one operation, through pointers so both sides do the same loads and the same
store and what is left is the computation (`clang -O2`, Apple M4 Pro):

| | FP32×4 | FP32×8 | FP32×16 | SI32×4 | SI16×8 | SI8×16 |
|---|---|---|---|---|---|---|
| `to_bits` | 17 → **10** | 33 → **15** | 65 → **27** | 17 → **10** | 39 → **10** | 71 → **14** |
| `permute` | 35 → **12** | 44 → **22** | 70 → **69** | 36 → **12** | 54 → **12** | 30 → **11** |
| `select` | 7 → 7 | 9 → 9 | 17 → 17 | 18 → **7** | 54 → **7** | 59 → **7** |
| `fma` | 7 → **6** | 9 → **7** | 30 → **13** | 14 → **6** | 54 → **6** | 30 → **6** |
| `add` | 5 → 5 | 6 → 6 | 11 → 11 | 12 → **5** | 38 → **5** | 27 → **5** |
| `min` | 6 → **5** | 8 → **6** | 15 → **11** | 16 → **5** | 46 → **5** | 27 → **5** |
| `sum` | 5 → 5 | 8 → **6** | 14 → **9** | 4 → 4 | 4 → 4 | 4 → 4 |

The float columns were already respectable before any backend existed — clang's SLP vectorizer
reassembles a lane loop over a plain array into `fadd.4s` on its own. The integer and narrow types
are where the table earns its keep, and `to_bits` and `permute` are where it earns it everywhere.

The rank grid, at every (operation, type, width) — `tests/test_arm_dispatch.cpp` prints it and
`require_at_least` asserts the floor, so a lost registration stops the **build**:

```
  op \ (type,lanes)   f,4   f,8  f,16   d,2   d,4   d,8   i,4   i,8  i,16
  fma                 20    10    10    20    10    10    20    10    10
  permute             20    10    10     0     0     0    20    10    10
  bcast_lane<1>       20    10    10    20    10    10    20    10    10
  cmp_gt              20    10    10    20    10    10    20    10    10
  (0 = GENERIC, a scalar loop; 10 = SPLIT; 20 = REGISTER; 30 = MASK_REGISTER)
```

63 of 63 cells were generic before; 3 are now, and they are one cell three times: `permute` on
`double`. `permute`'s index vector is always `SimdVec<SI32,N>`, and at two lanes that is 64 bits
of a 128-bit register — a width with no register impl of its own, so there is no `idx.data.reg` to
build a `TBL` control from. x86 leaves the identical cell generic for the identical reason. The
4- and 8-lane cases follow from it rather than being separate holes: the split form asks the half
whether it has anything better than a lane loop, and it has not.

**`SPLIT` is no longer a declared-and-unused rank.** It is the rank most cells on this
architecture reach, which is what pushed three gaps in it into view — `bcast_lane` and
`mask_from_bits` had no split form at all, and both got one; they improve an SSE-only x86 build
too.

**The AArch64 ABI needed no layout work at one register, and above one register no ABI can help.**
AAPCS64 passes a Homogeneous Vector Aggregate — one to four members, all the same vector type — in
`v0`-`v7`, so at `SimdSize<T>` lanes a value crosses a call in its register and the union that cost
a factor of two on SysV (§ 3) costs nothing here.

Above that width both conventions give up, and it took a red CI run to state it properly. SysV
sends the value to the **stack** — two `__m256d` classify SSE,SSEUP,SSEUP,SSEUP then SSE again, and
the fifth eightbyte alone is enough, which is exactly what § 3 calls "unrecoverable" about `Split`.
AAPCS64 passes it **by reference** in `x0`-`x7` with an indirect return through `x8`. So
`SimdVec<double,8>` — 64 bytes — travels through memory on every target without a 64-byte register,
and no layout change can alter that.

Which means the ABI probe was asserting the wrong thing at those widths: it demanded zero stack
traffic from a value the ABI cannot keep in registers. It passed only on the AVX-512 machine it was
written on, where eight doubles *are* one `zmm`, and failed on a CI runner with AVX2 for a reason
no code change could fix. The enforced set is now written at `SimdSize<T>` — one register by
construction, on every target — and the fixed-width probes are printed as information. What they
were really guarding, "the operation did not lose its register variant", is now a `require_at_least`
in both dispatch tests: a `static_assert` that names the operation instead of leaving an instruction
count to be interpreted.

The by-reference case also exposed a limit of the check itself: on AAPCS64 the callee reads its
argument through a pointer register and never touches `sp`, so the probe reported those widths
**clean** when the value was in memory all along. A green light that means nothing — the same
failure mode as `no_vecext.sh` in § 6. Enforcing only at one register removes the question.

### The one place the backend costs instructions

Integer `div`. Same number of `sdiv`s either way — neither ARM nor x86 has a SIMD integer division
— but with a register impl `values` is a vector type, and the compiler lowers a vector integer
division by extracting each lane, dividing, and inserting: 20 instructions against 12 at `SI32×4`.
`no_vecext.sh` already carried a note about this cell on x86; on ARM the sign is reversed, and
there is no instruction to dispatch to that would fix either.

One more honest number. `sum` in `bench_ops.cpp` went from 0.51 to 0.65 ns/op at the native width,
and it is **not** `ADDV` being slow — removing the `ADDV` form changes nothing (0.65 either way).
The generic form is plain code, so clang **unrolled the benchmark loop** eightfold and hid the
latency of the dependency chain; it cannot unroll through an opaque intrinsic. Fewer instructions
per call, less freedom for the optimiser around it. That is the trade every intrinsic makes, and
this is the one loop shape where it shows.

## 5. Building and testing

```sh
cd tests && make          # build and run the tests, for this machine
make isa                  # ... and at every feature level of THIS architecture, plus two MSVC-path ones
make matrix               # the (operation, type, ISA) compile map, 544 translation units
make msvc                 # what the dispatch table is worth WITHOUT the compiler's vector types
make lattice              # is the ARM feature lattice where ArmCpuFeatures.h says it is
make bench                # timings for the cross-lane operations
```

`make isa` and `make matrix` read the feature levels off the compiler's own `-dumpmachine`, not
off `uname`: the compiler may be a cross compiler, and `-msse2` handed to an aarch64 one is not a
lower feature level but a hard error. So the same two scripts drive x86-64 (SSE2 → SSE4.2 → AVX →
AVX2 → native) and AArch64 (ARMv8-A → ARMv8.2+fp16 → ARMv8.4+dotprod → native) with nothing
passed in.

On Windows, `pwsh tests/run_msvc.ps1` does what `make isa` does, at MSVC's four `/arch:` levels.
`.github/workflows/ci.yml` runs all of it — gcc and clang on x86-64 Linux, gcc and clang on a
**native arm64 runner**, and MSVC on Windows. The ARM job runs the tests rather than only building
them, which matters: this backend's failure modes are wrong *values* — a byte index off by one in
a `TBL` control, a `to_bits` reduction that overflows a `uint8_t` and silently drops the top eight
lanes — and a compile-only job would have caught neither.

`make isa` is the one that matters when changing a backend: `make` alone builds for
`-march=native`, so on a recent x86 box the SSE2 and AVX paths — the ones most likely to rot — are
never exercised, and on an ARM one the ARMv8-A baseline never is. It is what caught the last two
bugs of the audit, one of them in the test suite itself.

The build is described in `tests/xmake.lua`; the `Makefile` next to it is a thin wrapper that
calls xmake, so `make` works for anyone who would rather not learn a new tool.

| test | what it checks |
|---|---|
| `test_ops` | every operation, value by value |
| `test_split` | `SimdVec<float,8>` on an architecture capped at SSE2 — **the differentiator** |
| `test_selection` | total order, independence from declaration order, the safety net |
| `test_x86_ops` | **665 assertions.** A grid drives every operation over 23 (type, width) pairs — FP32/FP64/SI32/PI32/SI64/PI64/SI16 from 2 to 32 lanes, plus widths that are not a power of two and one wider than any register. Each pair lands on a different variant per target, and all of them must agree |
| `test_x86_dispatch` | the (operation, type, width) rank grid: which variant is *actually* selected, everywhere rather than at one width — and `require_at_least` on each cell that must have one, so a lost registration fails the **build** |
| `test_arm_ops` | **1 136 assertions.** The same grid over the ARM types and widths — including the 16- and 8-bit lanes, which Advanced SIMD covers uniformly and SSE does not — *and then the whole grid again over `ArmCpu<64,NEON,FMA>`*, the ARMv7-A floor, on whatever host it is compiled on. Everything A64-only has to disappear from under it and leave the **same answers** behind. Plus the operations x86 has no cell for: the per-lane variable shift and the integer multiply-accumulate |
| `test_arm_dispatch` | the rank grid and the floor, twice — at this target and at the ARMv7 floor, which is what says the two feature levels are really separate. And the **`SPLIT` floor**, which on ARM is the one that matters: every width above four floats must reach it, or the library is going through the stack for a value that fits in two registers |

Both backends' tests are built on both architectures, deliberately: `test_x86_ops.cpp` names
`X86Cpu<64,SSE2,SSE>` explicitly, so on an ARM host it still checks that an architecture with no
register impls gives the right answers through the generic forms — and its grid runs on
`NativeCpu`, i.e. on NEON. The same holds the other way. Each file is a value test everywhere and
a backend test on its own target.

and three checks that are not binaries:

`tests/compile_matrix.sh` builds one translation unit per (operation, type, ISA) across the four
feature levels of whatever the compiler targets — 544 of them — and prints the map, because a
coverage hole that is a *compile* error cannot be seen from a test binary. `SI16` and `SI8` joined
the type list for the ARM port and immediately paid for themselves: they are the types where a
hole is most likely, and they found one (§ 6).

`tests/run_all_isa.sh` builds and runs every test at every feature level of the compiler's target.
This is the one that matters when touching a backend: an intrinsic guarded by the wrong feature
macro compiles fine on a machine that has everything and not at all on the target that needs it.

`tests/check_arm_lattice.sh` checks the ARM feature boundary from both sides — see § 4. It needs
clang and nothing else.

And a check that is not an assertion but a **reading of the disassembly** — it runs on every
build, because the regression it catches shows up nowhere else:

```
ABI probe (arm64): a value that IS one register must cross a call in it
  probe_nat_fma         3 instructions, 0 touching sp
  probe_nat_fma_f64     3 instructions, 0 touching sp
  probe_nat_perm        9 instructions, 0 touching sp
  probe_nat_add         2 instructions, 0 touching sp
  probe_nat_sel         2 instructions, 0 touching sp
  probe_nat_sel_f64     2 instructions, 0 touching sp
ok: vectors and masks cross a call in their registers.
  above one register (both ABIs pass these through memory -- informational):
  probe_fma             7 instructions, 0 touching sp
  probe_fma_f64        13 instructions, 0 touching sp
  probe_perm           22 instructions, 0 touching sp
  probe_sel_lane        7 instructions, 0 touching sp
  probe_sel_bits       29 instructions, 0 touching sp
```

Six symbols enforced: any one of them touching the stack pointer fails the build and names the
cause. A `static_assert` can do nothing about an ABI, and the regression changes no result, only
the speed. Widening it from one probe to five was worth it immediately — two came out dirty, for
two *different* reasons: `probe_sel_lane` was the mask ABI of § 3, and `probe_fma_f64` was not an
ABI problem at all but a missing register variant at eight lanes of double, showing up in the same
measurement.

Every probe in the enforced set is written at `SimdSize<T>` and its mask flavour is *deduced* from
what `gt` returns on the target — four lanes and a `__m128i` under SSE2, eight and a `__m256i`
under AVX2, sixteen and a `__mmask16` under AVX-512, four and a `uint32x4_t` on ARM. Nothing in it
names a width or a flavour, so it is one register everywhere by construction rather than by luck.
That is the § 4 lesson: spelling `SimdMask<16,32>` asked AVX-512 for a 64-byte *lane* mask, a
flavour it never produces.

The ARM port also made the check **portable**, which it was not: the stack pointer is `sp` and not
`%rsp`, Mach-O prefixes symbols with an underscore, `--disassemble=` is a GNU binutils spelling
that LLVM's objdump rejects, and a frame pointer puts `%rsp` in a *clean* prologue (hence
`-fomit-frame-pointer`). Left alone the probe would have passed on ARM by never matching
anything.

## 6. Status

Working: **x86** (SSE2 / SSE4.1 / SSE4.2 / AVX / AVX2 / FMA / AVX-512VL) and **ARM** (NEON on
ARMv7-A, Advanced SIMD on AArch64), the recursive split for widths beyond the register, the
operations above, variant selection.

An audit went over the x86 side — by compiling and running it, not by reading it — and the results
are in [FINDINGS.md](FINDINGS.md). Where it stood, and where it stands:

| | before | after |
|---|---|---|
| architectures with a backend | 1 | **2** |
| (operation, type, ISA) cells that compile | 258 / 408 | **544 / 544** |
| (operation, type, width) cells with a register form, x86 | 8 / 63 | **59 / 63** |
| … ARM, reaching `REGISTER` or `SPLIT` rather than a lane loop | 0 / 63 | **60 / 63** |
| operations returning wrong values | 5 | **0** |
| values passed through memory across a call, at a width that IS one register | 2 of 5 probes | **0 of 6 enforced probes**, on 5 x86 levels and on ARM |
| value assertions, per feature level | 29 (one level) | **1 833**, at 5 x86 levels and 6 ARM ones |
| cells needing the compiler's vector extensions to be fast (the MSVC gap) | 12 / 168 | **4 / 196** on x86, all integer division; **2 / 196** on ARM, both a constant-folding artefact |
| compilers the suite runs under | 1 | **gcc 13, gcc 15, clang, MSVC** — all four now RUN it, not just build it |
| MSVC `/arch:` levels that compile, via the no-vector-extension path | — | **4 / 4** |

### What the ARM port found in the x86 code

Five bugs, none of them ARM's, and each found by a different mechanism — which is the argument for
a second backend that the performance numbers do not make:

1. **`any( a > b )` did not compile at 16 lanes** on any target with a register form at 8 — i.e.
   `-mavx` and up. The split branch of `NAME##_as_a_simd_mask` assumed both halves returned the
   *bit* flavour of mask, and a register form returns the *lane* flavour. It survived because
   nothing reached it: `to_bits( a > b )` and `select( a > b, … )` both route through
   `ops::cmp_gt`, and only `any`/`all` on a lazy comparison use that function. Found by writing
   the ARM grid, where the same shape occurs at `SimdVec<SI16,16>` — an ordinary width.

2. **`FP32×4` and `FP64×2` were registered twice** on any AVX target — once by SSE2 (`cmpps`) and
   once by AVX (`vcmpps`, three operands and a predicate). These are ordinary function overloads,
   not ranked variants, so both constraints held, neither subsumed the other, and the call was
   *ambiguous*. Exactly the situation `Selection.h`'s KNOWN LIMITS note describes; fixed with the
   remedy it prescribes, an explicit exclusion.

3. **`iota( beg, mul )` did not compile on the 8- and 16-bit lane types** at any width that
   splits. `beg + n * mul` is subject to the integer promotions, so on a 16-bit lane the recursive
   call became `iota( int, short, … )` and `T` could not be deduced. Found by adding `SI16` and
   `SI8` to the compile matrix. That call now has a cell in both grids.

4. **`a - b` did not compile without the compiler's vector extensions**, at a width where a
   register impl exists but the operation has no register form — 256-bit integers, i.e. MSVC's
   `/arch:AVX`. The generic form's `requires { a.data.values OP b.data.values; }` is satisfied by
   **array-to-pointer decay**: `ptr - ptr` is valid and yields a `ptrdiff_t`, which then will not
   assign back to an array. `+`, `*`, `/` and `&` are ill-formed on pointers, so `sub` was the only
   operator affected. The test now asks whether the result can be *assigned back*, which is what it
   always meant. Found by MSVC's first CI run; no local row reached it, because the two `MSVC path`
   rows were SSE2 and native and the gap is precisely between them. There is an `-mavx` one now.

5. **`no_vecext.sh` had been measuring nothing** on any host whose `objdump` is LLVM's, or whose
   object format is Mach-O: `--disassemble=probe` fails, `grep -c` on empty output is 0, and every
   cell read `0->0`. It reported "0 of 168 cells degrade" — a clean sweep that was a broken tool.
   With it fixed, four cells degrade on x86 (integer division, which has no instruction on either
   architecture) and two on ARM (`iota` with a compile-time argument, where both builds fold the
   constant and what is being counted is how the compiler chooses to materialise sixteen bytes).
   Both are now written down in the script's own closing note. It also showed a real hole while it
   was at it: 64-bit integer `min`/`max`, which ARM has no instruction for, was leaning on the
   compiler — `CMGT` plus `BSL` now does it in two.

Two gaps in the `SPLIT` rank turned up the same way, and closing them improves an SSE-only x86
build as much as ARM: `bcast_lane` and `mask_from_bits` had no split form at all.

And one in `Selection.h` itself, found by CI rather than by either backend. **`sel::search` was a
`constexpr` function template, and gcc 13 rejected an instantiation of it** — "used before its
definition", for a definition eight lines above the use, which is an instantiation-ordering
complaint about a chain the split ranks make deep on purpose. It is a class template now, so the
diagnostic is inexpressible; and that is this file's own argument applied to the one piece of the
mechanism that was not following it (§ 2: *a variant is a class specialization, never a function
overload, because specializations are looked up at the point of instantiation*). The property the
new form has to preserve — that a rank below the selected one is **never instantiated** — was
being relied on and checked by nothing, and is now pinned by a variant whose `available` is a
hard error if it is ever looked at. See FINDINGS § 9.5.

Next up:

- **SVE.** The declaration is in place and honest — no width, because the length is not a
  compile-time constant. `-msve-vector-bits=N` makes the SVE types sized, at which point they can
  be union members like any other and an `SVE<N>` feature carrying a width is exactly right. It is
  also where the `MASK_REGISTER` rank finally means something on ARM: SVE has real predicate
  registers, and `to_bits` stops being a reduction.
- **`FP16` and `BF16` want a lane type, not a backend.** `common_types.h` has no 16-bit float, so
  there is no `SimdVec<FP16,8>` for a variant to key on — and eight half-precision lanes in 128
  bits is the widest ARM gets. That is a `common_types.h` change first.
- **`DOTPROD` and `I8MM` want a new operation.** `SDOT` accumulates four 8-bit products into each
  32-bit lane; it is a *reduction*, so it is not a variant of `fma` but an operation of its own.
  x86 has nothing below AVX-512-VNNI to match it.
- **MSVC now runs, and it is green** — the last thing on this list to become true. All four
  `/arch:` levels build and the three that can be executed on a hosted runner pass every
  assertion. It took three rounds and each one found something: a compile error from
  array-to-pointer decay (§ 6), a runtime failure, and then the discovery that **both test
  harnesses were filtering out the diagnostic that would have explained it** — and swallowing
  `check.h`'s `XPASS` along with it.

  The runtime failure was on a store round trip at `/arch:AVX2` only, and it is closed without
  ever having been reproduced off Windows: it went away once the check's own two output buffers
  stopped asking for `alignas( 64 )` they never needed — they feed `store_unaligned`. Which of the
  three changes made in that round actually did it is not established, and is written down as not
  established. The library was never implicated: `SimdVec<SI16,16>` has no register impl on x86
  below AVX-512BW, so `/arch:AVX` and `/arch:AVX2` take the byte-identical splittable path, and
  the same lanes are read by fifteen other passing assertions in the same cell. See FINDINGS § 9.6.

- **MSVC's ARM64 target is still unverified**, and is the honest remainder: `<arm64_neon.h>`
  rather than `<arm_neon.h>`, `__prefetch` rather than `__builtin_prefetch`, and none of the
  `__ARM_FEATURE_*` macros defined at all, so `arm_intrin.h` takes what the architecture
  guarantees and nothing more. Reasoned from documented behaviour, not measured — adding a leg
  for it needs a cross build and a runner to execute on.

- **A two-lane `permute` of 64-bit elements** would close the last three generic cells on ARM by
  unlocking the split at 4 and 8 lanes. It is buildable — narrow the index, `TBL` — and whether it
  beats a memory round trip at two lanes is unmeasured. Guessing is what `available` exists to
  prevent.
