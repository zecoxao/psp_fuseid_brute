# psp_nand_fuseid_brute - GPU (NVIDIA / CUDA Driver API) build
#
# No CUDA toolkit required:
#   * the device kernel is compiled to PTX by clang's built-in NVPTX backend
#   * the host links only against the driver (libcuda) shipped with the GPU
#   * the PTX is JIT-compiled by the driver at run time, so one build runs on
#     any NVIDIA GPU (Ampere RTX 3050 ... Blackwell RTX 5060 and newer)

CC       ?= gcc
CLANG    ?= clang
CFLAGS   ?= -O2 -Wall -D_FILE_OFFSET_BITS=64
# sm_52 virtual target -> forward-compatible PTX the driver JITs to any newer GPU
ARCH     ?= sm_52

# CUDA driver library location (WSL ships it here; native Linux: /usr/lib etc.)
CUDA_LIBDIR ?= /usr/lib/wsl/lib
LDFLAGS  ?= -L$(CUDA_LIBDIR) -Wl,-rpath,$(CUDA_LIBDIR) -lcuda

TARGET = psp_nand_fuseid_brute
PTX    = kernel.ptx

all: $(TARGET) $(PTX)

$(PTX): kernel.c
	$(CLANG) --target=nvptx64-nvidia-cuda -march=$(ARCH) -O3 -S -o $@ $<

$(TARGET): main.c cuda_min.h
	$(CC) $(CFLAGS) -o $@ main.c $(LDFLAGS)

clean:
	rm -f $(TARGET) $(PTX)

.PHONY: all clean
