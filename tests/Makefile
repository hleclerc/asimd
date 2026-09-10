# A thin wrapper around xmake, for people who would rather type `make`.
#
#   make          build everything and run the tests
#   make build
#   make run
#   make clean
#
# xmake IS the build system here -- see `xmake.lua`, which is where the flags, the target list and
# the ABI check actually live. Installing it is a one-liner:
#
#     curl -fsSL https://xmake.io/shget.text | bash        # or: apt / brew / pacman install xmake
#
# The `-P .` matters when this directory sits inside a larger project: without it xmake walks up
# and picks the parent's `xmake.lua`.

XMAKE ?= xmake
TESTS := test_ops test_split test_selection test_x86_ops test_x86_dispatch

all: run

build:
	@command -v $(XMAKE) >/dev/null 2>&1 || { \
	    echo "xmake not found. Install it with:"; \
	    echo "    curl -fsSL https://xmake.io/shget.text | bash"; \
	    exit 1; }
	@$(XMAKE) f -P . -m release -y
	@$(XMAKE) -P .

run: build
	@for t in $(TESTS); do $(XMAKE) run -P . $$t || exit 1; done

# every test at every x86 feature level, straight through the compiler. `make` alone only ever
# builds for this machine, which on a recent box means the SSE2 and AVX paths are never exercised.
isa:
	@./run_all_isa.sh $(CXX_FOR_ISA)

# the (operation, type, ISA) compile map. A coverage hole that is a *compile* error cannot be
# seen from a test binary.
matrix:
	@./compile_matrix.sh $(CXX_FOR_ISA)

# WHAT THE DISPATCH TABLE IS WORTH ON ITS OWN. Compiles every cell twice, with and without the
# compiler's vector extensions -- the second being exactly MSVC's situation, since it has none.
# A cell that changes is a cell gcc was carrying, i.e. a hole in the table.
msvc:
	@./no_vecext.sh "$(ARCH)" $(CXX_FOR_ISA)

# TIMINGS for the cross-lane operations, where counting instructions is a poor proxy: the generic
# `permute` goes through the stack, and the reload is a store-to-load forwarding stall that no
# instruction count shows.
bench:
	@$(CXX_FOR_ISA) -std=c++20 -O2 $(ARCH) -I../src -I. bench_ops.cpp -o /tmp/asimd_bench && /tmp/asimd_bench

clean:
	@$(XMAKE) clean -P . >/dev/null 2>&1 || true
	@rm -rf build .xmake

CXX_FOR_ISA ?= g++
ARCH        ?= -march=native

.PHONY: all build run isa matrix msvc bench clean
