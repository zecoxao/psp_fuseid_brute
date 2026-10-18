# psp_nand_fuseid_brute

GPU brute-forcer that recovers a PSP console's **Tachyon fuse id** from a NAND
dump, ported from the reference NAND extractor (`../main.c`, `nand.c`,
`lflash.c`, `sysreg.c`).

## How it works

The fuse id only enters a NAND through the **lflash scramble** seed
(`pspLfatfsGenSeedx1`):

```
seed(flash n) = low32(fuseid) ^ ror(high32(fuseid), 0xFE*n)
```

This is linear in the two fuse-id halves, so the full id is recoverable from a
**scrambled** dump (FW 3.00+):

1. **Phase 2** — brute the 32-bit flash0 seed (`= low32 ^ high32`) against the
   flash0 PBR's 128-bit FAT boot signature (`00*14 55 AA` at 0x1F0). 2³²
   candidates, run on the GPU.
2. **Phase 3** — brute the high half `B`; for each candidate reconstruct the
   other scrambled flashes' seeds (flash2/flash3) and require their PBR
   signatures too. Two extra flashes pin `B` uniquely →
   `fuseid = (B<<32) | (seed0 ^ B)`.

If the dump's logic data is **not scrambled** (early firmware, or tachyon
console where lflash is plain), the fuse id leaves no footprint and the tool
says so instead of guessing.

## Build (WSL / Linux, no CUDA toolkit)

```
make
```

* The device kernel (`kernel.c`) is compiled to `kernel.ptx` by **clang's NVPTX
  backend** — no CUDA toolkit, no nvcc.
* The host links only against the **driver** (`libcuda`, shipped with the GPU).
* The PTX is JIT-compiled by the driver at run time, so one build runs on any
  NVIDIA GPU (Ampere RTX 3050 … Blackwell RTX 5060 and newer).

Override the virtual arch with `make ARCH=sm_75` and the driver dir with
`make CUDA_LIBDIR=/usr/lib/x86_64-linux-gnu` on native Linux.

## Usage

```
./psp_nand_fuseid_brute <nand_dump> [tachyon] [fuseid_hint]
```

`tachyon` is metadata only (printed, not used for recovery; default 0x200000).
A `fuseid_hint` positional arg is accepted and ignored — it is what the tool
recovers.

Options:

| flag | meaning |
|------|---------|
| `--cpu`          | brute on CPU (1 thread) instead of GPU |
| `--ptx <path>`   | explicit path to `kernel.ptx` |
| `--start <hex>`  | phase-2 start seed (default 0) |
| `--count <hex>`  | phase-2 seed count (default 0x100000000) |
| `--range3 <hex>` | phase-3 high-half search size (default 0x1000000) |

### Verified example

```
$ ./psp_nand_fuseid_brute ../9g_nand.bin 0x8200000
 Recovered Fuse Id : 0x353C8C0AAF90          (~2 s on an RTX 3050 Laptop)
   low32  = 0x8C0AAF90   high32 = 0x0000353C
   flash0 seed = 0x8C0A9AAC
   flash2 seed = 0x8C09FC50
   flash3 seed = 0x8C07E090
```
