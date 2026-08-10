CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wextra
LDFLAGS  = -lm -lpthread

# Native arm64 builds need to compile for the SIMD that doe.c already guards.
# Stock aarch64 targets can omit __ARM_FEATURE_DOTPROD unless the target arch is
# explicit, which leaves the int8 packed matvec kernels compiled out.
ARM_SIMD ?= 1
ifneq ($(filter aarch64 arm64,$(shell uname -m)),)
  ifeq ($(ARM_SIMD),1)
    ARM_HAS_DOT := $(shell echo | $(CC) -dM -E - 2>/dev/null | grep -c __ARM_FEATURE_DOTPROD)
    ifeq ($(ARM_HAS_DOT),0)
      ARM_MARCH := $(shell f=""; \
        { grep -qwm1 asimddp /proc/cpuinfo 2>/dev/null || \
          [ "$$(sysctl -n hw.optional.arm.FEAT_DotProd 2>/dev/null)" = 1 ]; } && f="$$f+dotprod"; \
        { grep -qwm1 i8mm /proc/cpuinfo 2>/dev/null || \
          [ "$$(sysctl -n hw.optional.arm.FEAT_I8MM 2>/dev/null)" = 1 ]; } && f="$$f+i8mm"; \
        [ -n "$$f" ] && echo "-march=armv8.2-a$$f")
      ARM_FLAGS ?= $(shell [ -n "$(ARM_MARCH)" ] && \
        $(CC) $(ARM_MARCH) -E -x c /dev/null >/dev/null 2>&1 && echo "$(ARM_MARCH)")
      CFLAGS += $(ARM_FLAGS)
      ARM_NAME = $(if $(ARM_FLAGS),$(ARM_FLAGS),baseline armv8-a)
    else
      ARM_NAME = compiler default (already carries SDOT)
    endif
  endif
endif
ARM_NAME ?= disabled or non-arm64

MM      ?= clang++
MMFLAGS ?= -O2 -Wall -Wextra -std=c++17 -I.

.PHONY: all arm-info clean blas openblas metal cuda

all: doe_field

arm-info:
	@printf 'ARM SIMD: %s\n' "$(ARM_NAME)"
	@printf 'CFLAGS: %s\n' "$(CFLAGS)"

doe_field: doe.c
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

blas: doe.c gguf.c
	$(CC) $(CFLAGS) -DUSE_BLAS -DACCELERATE doe.c gguf.c $(LDFLAGS) -framework Accelerate -o doe_field

openblas: doe.c gguf.c
	$(CC) $(CFLAGS) -DUSE_BLAS doe.c gguf.c $(LDFLAGS) -lopenblas -o doe_field

notorch_metal.o: notorch_metal.mm notorch_metal.h
	$(MM) $(MMFLAGS) -DUSE_METAL -fobjc-arc -c notorch_metal.mm -o notorch_metal.o

metal: doe.c gguf.c notorch_metal.o notorch_metal.h
	$(CC) $(CFLAGS) -DUSE_BLAS -DACCELERATE -DUSE_METAL doe.c gguf.c notorch_metal.o $(LDFLAGS) -framework Accelerate -framework Metal -framework Foundation -lc++ -o doe_field

cuda: doe.c
	$(CC) $(CFLAGS) -DUSE_CUBLAS $< $(LDFLAGS) -lcublas -lcudart -o doe_field

clean:
	rm -f doe_field notorch_metal.o
