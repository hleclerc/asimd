-- asimd's tests. Self-contained: `xmake -P . && xmake run -P . test_ops`.
--
-- The `-P .` is not decorative when this lives inside a larger project: without it xmake walks up
-- and picks the parent's `xmake.lua`.
--
-- A `Makefile` sits next to this file; it is a thin wrapper that calls xmake, for anyone who
-- would rather type `make`. The build itself is described here and only here.
--
-- FOR THE ARM PORT, what is missing on a typical x86 box installs with:
--
--     sudo apt install g++-aarch64-linux-gnu qemu-user
--     xmake f -P . -p linux -a arm64 --toolchain=gcc --sdk=/usr && xmake -P .
--
-- and the binaries run under `qemu-aarch64 -L /usr/aarch64-linux-gnu`. `zig cc` does the same
-- without installing a cross toolchain (`xmake f -P . --toolchain=zig`), if you prefer a single
-- download.

set_project( "asimd_tests" )
set_languages( "c++20" )
add_rules( "mode.debug", "mode.release" )

-- header-only library: one include directory, nothing else.
add_includedirs( "../src" )
set_warnings( "all" )

-- THE TARGET ARCHITECTURE IS AN OPTION, not a constant. `-march=native` was hard-coded here,
-- which contradicts the cross-build recipe in the header comment above and makes it impossible
-- to check that the SSE2 or AVX paths still compile -- the ones most likely to rot, since the
-- machine you develop on has everything.
--
--     xmake f -P . --arch_flags="-mavx2 -mfma"
--
-- `tests/run_all_isa.sh` does the same thing with the compiler directly, for all five levels at
-- once, and is the quicker way to answer the question.
option( "arch_flags" )
    set_default( "-march=native" )
    set_showmenu( true )
    set_description( "compiler flags selecting the target ISA (e.g. -msse2, -mavx2 -mfma)" )
option_end()

if is_mode( "release" ) then
    add_cxflags( "-O3", { force = true } )
    for _, f in ipairs( ( get_config( "arch_flags" ) or "-march=native" ):split( "%s+" ) ) do
        add_cxflags( f, { force = true } )
    end
    add_defines( "NDEBUG" )
end

-- The upstream tests (`test_SimdVec.cpp` & co) need Catch2, which nothing else here requires: they
-- are kept but not built. Hence an explicit list rather than a glob.
-- BOTH BACKENDS' TESTS ARE BUILT ON BOTH ARCHITECTURES, and that is deliberate rather than
-- wasteful. `test_x86_ops.cpp` names `X86Cpu<64,SSE2,SSE>` explicitly, so on an ARM host it
-- still checks that an architecture with no register impls gives the right answers through the
-- generic forms -- and its grid runs on `NativeCpu`, i.e. on NEON. The same holds the other way:
-- `test_arm_ops.cpp` on an x86 host exercises `ArmCpu<64,NEON,FMA>` with nothing registered
-- under it. Each file is a value test everywhere and a backend test on its own target.
for _, name in ipairs( { "test_ops", "test_split", "test_selection", "test_rotate",
                         "test_x86_ops", "test_x86_dispatch",
                         "test_arm_ops", "test_arm_dispatch" } ) do
    target( name )
        set_kind( "binary" )
        add_files( name .. ".cpp" )
end

-- ---------------------------------------------------------------------------------------------
-- THE CHECK THAT IS NOT AN ASSERTION, and that runs on EVERY build.
--
-- A `static_assert` can do nothing about an ABI. The layout of `SimdVecImpl` decides whether a
-- vector crosses a call in a register or through memory; getting it wrong cost a factor of two
-- here, with no test failing and no result changing. So we read the disassembly and count stack
-- accesses: zero expected. See § 3 of the README.
target( "abi_probe" )
    set_kind( "object" )
    add_files( "abi_probe.cpp" )
    -- NO FRAME POINTER. `movq %rsp, %rbp` in a prologue is not a value going through memory, but
    -- it does touch `%rsp` -- so on any target that keeps a frame pointer by default (macOS, and
    -- `-O0`) every probe read "1 touching %rsp" and the check either had to allow one, and miss a
    -- real single spill, or fail on all of them. Removing the prologue makes zero mean zero.
    add_cxflags( "-fomit-frame-pointer", { force = true } )
    -- `extern "C"` is what gives these probes stable symbol names for `objdump` to find; clang
    -- then warns that a C-linkage function returns a C++ type, on every one of them, on every
    -- build. Scoped to clang because gcc has no such warning name and rejects unknown ones once
    -- anything else goes wrong.
    add_cxflags( "-Wno-return-type-c-linkage", { force = true, tools = { "clang", "clangxx" } } )
    after_build( function ( target )
        import( "core.base.option" )
        local obj = target:objectfiles()[ 1 ]

        -- WHAT IS ENFORCED, AND WHY THE REST IS NOT. See the header comment of `abi_probe.cpp`:
        -- the ABI question only has a right answer at a width that IS one register, and the
        -- `probe_nat_*` probes are written at `SimdSize<T>` precisely so that they are that width
        -- on every target -- four lanes under SSE2 and on any ARM part, eight under AVX2, sixteen
        -- under AVX-512.
        --
        -- Above one register NEITHER convention can pass the value in registers: SysV sends it to
        -- the stack (the fifth eightbyte of two `__m256d` is SSE, not SSEUP) and AAPCS64 passes it
        -- BY REFERENCE in `x0`-`x7` with an indirect return through `x8`. Enforcing zero stack
        -- traffic there asserted something no code change can deliver -- and it did fail CI, on
        -- `probe_fma_f64`, which is 64 bytes and therefore one register only on AVX-512.
        local must_be_clean = { "probe_nat_fma", "probe_nat_fma_f64", "probe_nat_perm",
                                "probe_nat_add", "probe_nat_sel", "probe_nat_sel_f64" }
        -- A KNOWN LIMIT OF THE ZERO-STACK TEST, written down rather than papered over: on
        -- AAPCS64 a composite of 16 bytes or less that is NOT a Homogeneous Vector Aggregate is
        -- passed in `x0`-`x7`, not on the stack -- so a mask impl that regrew an array would
        -- arrive in general registers and this count would still read zero. What catches that
        -- there is `require_at_least` in the dispatch tests, which is a static_assert and does
        -- not depend on reading a disassembly at all.
        --
        -- disassembled and printed, not enforced: one register on AVX2 and up, two or more
        -- elsewhere. What they show is whether the SPLIT path keeps register-level operations
        -- underneath it, which `require_at_least` in the dispatch tests asserts properly.
        local informational  = { "probe_fma", "probe_fma_f64", "probe_perm",
                                 "probe_sel_lane", "probe_sel_bits" }

        -- THREE THINGS DIFFER BETWEEN HOSTS, and all three broke this check on ARM.
        --
        --   THE STACK POINTER IS NOT CALLED `%rsp`. It is `sp` on AArch64, so the pattern that
        --   detects a spill has to follow the architecture. Left as it was, the check would have
        --   passed on ARM by never matching anything -- a green light that means nothing, which
        --   is worse than no check.
        --
        --   MACH-O PREFIXES SYMBOLS WITH AN UNDERSCORE. `_probe_fma`, not `probe_fma`.
        --
        --   `--disassemble=` IS A GNU BINUTILS SPELLING. LLVM's objdump -- which is what
        --   `objdump` is on macOS -- wants `--disassemble-symbols=`. Both are tried.
        local arch = os.arch() or ""
        local is_arm  = arch:find( "arm" ) ~= nil or arch:find( "aarch64" ) ~= nil
        local stack_re = is_arm and "%f[%w]sp%f[%W]" or "%%rsp"
        local stack_nm = is_arm and "sp" or "%rsp"

        local function disassemble( sym )
            for _, name in ipairs( { sym, "_" .. sym } ) do
                for _, flag in ipairs( { "--disassemble-symbols=", "--disassemble=" } ) do
                    -- `try`, not `pcall`: xmake's sandbox does not expose `pcall`, and a wrong
                    -- flag or a missing symbol makes `objdump` exit non-zero, which `os.iorunv`
                    -- turns into an error rather than a return value.
                    local dis = try { function () return os.iorunv( "objdump", { "-d", flag .. name, obj } ) end }
                    if dis and dis:find( "%x+:" ) then return dis end
                end
            end
            return nil
        end

        local function count( sym )
            local dis = disassemble( sym )
            if not dis then return nil, nil end
            local n_stack, n_tot = 0, 0
            for line in dis:gmatch( "[^\n]+" ) do
                if line:match( "^%s+%x+:" ) then
                    n_tot = n_tot + 1
                    if line:match( stack_re ) then n_stack = n_stack + 1 end
                end
            end
            return n_tot, n_stack
        end

        local function measure( sym )
            local n_tot, n_stack = count( sym )
            if not n_tot then
                -- NOT SILENTLY OK. A probe that cannot be read is a probe that is not checking
                -- anything, and this whole target exists because a `static_assert` cannot see an
                -- ABI regression. `no_vecext.sh` spent its whole life reporting a clean sweep for
                -- exactly this reason -- see § 9.2 of FINDINGS.md.
                raise( sym .. ": could not disassemble it out of " .. obj .. ". `objdump` is "
                    .. "needed for this check; on macOS it comes with the command line tools." )
            end
            return n_tot, n_stack
        end

        cprint( "${bright}ABI probe${clear} (%s): a value that IS one register must cross a call in it", arch )
        for _, sym in ipairs( must_be_clean ) do
            local n_tot, n_stack = measure( sym )
            cprint( "  %-20s %2d instructions, %d touching %s", sym, n_tot, n_stack, stack_nm )
            if n_stack > 0 then
                raise( sym .. ": this value is passed through MEMORY, and at this width it fits "
                    .. "in ONE register -- so this is a layout regression, not the ABI. Either "
                    .. "the union in SIMD_VEC_IMPL_REG / SIMD_BOOL_IMPL_REG_LARGE has grown an "
                    .. "array or a Split again (README, section 3), or the operation lost its "
                    .. "register variant and fell back to the generic lane loop (FINDINGS.md, "
                    .. "section 3)." )
            end
        end
        cprint( "${green}ok${clear}: vectors and masks cross a call in their registers." )

        cprint( "${bright}  above one register${clear} (both ABIs pass these through memory -- "
             .. "informational, see abi_probe.cpp):" )
        for _, sym in ipairs( informational ) do
            local n_tot, n_stack = measure( sym )
            cprint( "  %-20s %2d instructions, %d touching %s", sym, n_tot, n_stack, stack_nm )
        end
    end )
