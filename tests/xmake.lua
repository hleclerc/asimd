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
for _, name in ipairs( { "test_ops", "test_split", "test_selection",
                         "test_x86_ops", "test_x86_dispatch" } ) do
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
    after_build( function ( target )
        import( "core.base.option" )
        local obj = target:objectfiles()[ 1 ]

        -- ALL FOUR MUST BE CLEAN NOW. Two of them were not when this probe was widened:
        -- `probe_fma_f64` measured 23 instructions with 12 stack accesses because no `fma` was
        -- registered at eight lanes of double, and `probe_sel_lane` 9 with 3 because
        -- SIMD_MASK_IMPL_REG_LARGE still held an array and a Split in its union -- the very
        -- layout README section 3 fixed for vectors, never applied to masks.
        local must_be_clean = { "probe_fma", "probe_perm", "probe_fma_f64", "probe_sel_lane",
                                "probe_sel_bits" }

        local function count( sym )
            local dis = os.iorunv( "objdump", { "-d", "--disassemble=" .. sym, obj } )
            local n_rsp, n_tot = 0, 0
            for line in dis:gmatch( "[^\n]+" ) do
                if line:match( "^%s+%x+:" ) then
                    n_tot = n_tot + 1
                    if line:match( "%%rsp" ) then n_rsp = n_rsp + 1 end
                end
            end
            return n_tot, n_rsp
        end

        cprint( "${bright}ABI probe${clear} (a value crossing a call by value):" )
        for _, sym in ipairs( must_be_clean ) do
            local n_tot, n_rsp = count( sym )
            cprint( "  %-16s %2d instructions, %d touching %%rsp", sym, n_tot, n_rsp )
            if n_rsp > 0 then
                raise( sym .. ": this value is passed through MEMORY. Either the union in "
                    .. "SIMD_VEC_IMPL_REG / SIMD_MASK_IMPL_REG_LARGE has grown an array or a "
                    .. "Split again (README, section 3), or the operation lost its register "
                    .. "variant and fell back to the generic lane loop (FINDINGS.md, section 3)." )
            end
        end
        cprint( "${green}ok${clear}: vectors and masks cross a call in their registers." )
    end )
