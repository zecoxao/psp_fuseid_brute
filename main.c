/*
 * psp_nand_fuseid_brute - recover a PSP console's Tachyon fuse id from a NAND
 * dump by brute-forcing the lflash scramble seed on the GPU.
 *
 * Ported from the reference NAND extractor (../main.c, nand.c, lflash.c,
 * sysreg.c).  The fuse id only ever enters a NAND through the lflash scramble:
 *
 *     seed(flash n) = low32(fuseid) ^ ror(high32(fuseid), 0xFE*n)   [GenSeedx1]
 *
 * which is *linear* in the two fuse-id halves, so:
 *
 *   Phase 2  brute the full 32-bit flash0 seed (= low32 ^ high32) against the
 *            flash0 PBR's 128-bit boot signature  ->  seed0.
 *   Phase 3  brute the high half B; for each candidate reconstruct the other
 *            scrambled flashes' seeds and require their PBR signatures too.
 *            With two extra flashes (e.g. flash2 & flash3) B is pinned
 *            uniquely, giving the full fuse id = (B<<32) | (seed0 ^ B).
 *
 * The heavy Phase 2 (2^32 candidates) runs on any NVIDIA GPU through the CUDA
 * Driver API + a clang-built PTX kernel (no CUDA toolkit needed).  A 1-thread
 * CPU fallback is available with --cpu.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include "cuda_min.h"

typedef uint32_t u32;
typedef uint64_t u64;

/* ---- NAND geometry (matches the reference extractor) ---- */
#define PAGE_SIZE   0x200u
#define EXTRA_SIZE  0x10u
#define PPB         0x20u          /* pages per block */
#define RAW_U8      PAGE_SIZE
#define PAD_U32     0x184u         /* padded device page: 0x610 bytes          */

#define SCRAMBLE_FLAG 0x0Du        /* 0b1101: flash0, flash2, flash3 scrambled */

static unsigned char *g_nand = NULL;
static u64  g_nand_size = 0;
static u32  g_total_pages = 0, g_total_blocks = 0, g_total_lblocks = 0;
static u32 *g_lpt = NULL;

/* ---------- bit helpers (match compat.h _rotr / nand.c bitrev) ---------- */
static inline u32 rotr(u32 v, int shift)
{
    unsigned s = (unsigned)shift & 31u;
    return s ? (v >> s) | (v << (32 - s)) : v;
}
static inline u32 bitrev(u32 x)
{
    x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
    x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
    x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
    x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
    x = ((x >> 16) & 0xffffu)    | ((x & 0xffffu) << 16);
    return x;
}

/* raw (scrambled) page bytes straight out of the dump */
static const unsigned char *raw_page(u32 ppn)
{
    return g_nand + (u64)ppn * (PAGE_SIZE + EXTRA_SIZE);
}
static const unsigned char *spare_of(u32 ppn)
{
    return g_nand + (u64)ppn * (PAGE_SIZE + EXTRA_SIZE) + PAGE_SIZE;
}

/* CPU descramble + PBR signature test (reference pspNandUnscramblePage) */
static int cpu_sig_ok(const u32 *raw /*PAD_U32*/, u32 ppn, u32 seed)
{
    u32 s   = rotr(seed, 0x15);
    u32 k   = rotr(ppn, 0x11) ^ ((s << 3) - s);
    u32 sel = 0x10u * ((ppn ^ s) & 0x1Fu);
    u32 e1, e2, e3, e4;
    for (int it = 0; it < 32; ++it) {
        u32 idx = sel >> 2;
        e1 = raw[idx]; e2 = raw[idx + 1]; e3 = raw[idx + 2]; e4 = raw[idx + 3];
        sel += 0x10u;
        e1 -= k; k += e1;
        e2 -= k; k ^= e2;
        e3 -= k; k -= e3;
        e4 -= k; k += e4;
        if (it == 31)
            return e1 == 0 && e2 == 0 && e3 == 0 && e4 == 0xAA550000u;
        sel &= 0x5FFu;
        k = bitrev(k + s);
    }
    return 0;
}

/* ---------- NAND / LPT / partitions ---------- */
static int load_nand(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Can't open %s\n", path); return -1; }
    fseeko(f, 0, SEEK_END);
    g_nand_size = (u64)ftello(f);
    fseeko(f, 0, SEEK_SET);
    g_nand = malloc(g_nand_size);
    if (!g_nand) { fclose(f); return -1; }
    if (fread(g_nand, 1, g_nand_size, f) != g_nand_size) { fclose(f); return -1; }
    fclose(f);
    g_total_pages  = (u32)(g_nand_size / (PAGE_SIZE + EXTRA_SIZE));
    g_total_blocks = g_total_pages / PPB;
    printf("NAND: 0x%llX bytes, 0x%X pages, 0x%X blocks\n",
           (unsigned long long)g_nand_size, g_total_pages, g_total_blocks);
    return 0;
}

static int build_lpt(void)
{
    if (g_total_blocks == 0x800)       g_total_lblocks = 0x780;
    else if (g_total_blocks == 0x1000) g_total_lblocks = 0xF00;
    else { fprintf(stderr, "Unknown NAND size (0x%X blocks)!\n", g_total_blocks); return -1; }

    g_lpt = malloc((size_t)g_total_blocks * sizeof(u32));
    if (!g_lpt) return -1;
    memset(g_lpt, 0xFF, (size_t)g_total_blocks * sizeof(u32));

    for (u32 pb = 0; pb < g_total_blocks; ++pb) {
        const unsigned char *sp = spare_of(pb * PPB);
        if (sp[5] != 0xFF) continue;                 /* bad block */
        if (*(const u32 *)&sp[8] == 0u) {
            u32 lb = (u32)sp[6] * 0x100u | sp[7];
            if (lb < g_total_blocks) g_lpt[lb] = pb;
        }
    }
    return 0;
}

/* physical page 0 of a logical block (or (u32)-1 if undefined) */
static u32 lblock_ppn0(u32 lbn)
{
    u32 pb = g_lpt[lbn];
    if (pb & 0x8000u) return (u32)-1;
    return pb * PPB;
}

typedef struct { int flashnum; u32 pbr_lbn; u32 ppn; } flash_t;

/* walk MBR/EBR (unscrambled) and collect every flash partition's PBR */
static int walk_flashes(flash_t *out, int max, int *n)
{
    *n = 0;
    u32 p = lblock_ppn0(0);
    if (p == (u32)-1) { fprintf(stderr, "MBR block undefined\n"); return -1; }
    const unsigned char *mbr = raw_page(p);
    u32 ext_rel = *(const u32 *)(mbr + 0x1BE + 8);
    u32 EBR0 = ext_rel / 0x20u, EBR = EBR0;
    int fn = 0;
    for (;;) {
        u32 ep = lblock_ppn0(EBR);
        if (ep == (u32)-1) { fprintf(stderr, "EBR lbn 0x%X undefined\n", EBR); return -1; }
        const unsigned char *ebr = raw_page(ep);
        u32 fat_rel  = *(const u32 *)(ebr + 0x1BE + 8);
        u32 next_rel = *(const u32 *)(ebr + 0x1CE + 8);
        u32 pbr_lbn  = EBR + fat_rel / 0x20u;
        if (*n < max) {
            out[*n].flashnum = fn;
            out[*n].pbr_lbn  = pbr_lbn;
            out[*n].ppn      = lblock_ppn0(pbr_lbn);
            (*n)++;
        }
        if (next_rel == 0) break;
        EBR = EBR0 + next_rel / 0x20u;
        fn++;
    }
    return 0;
}

/* raw PBR page padded to PAD_U32 (real bytes + zero pad) for the kernels */
static void load_padded_page(u32 *dst /*PAD_U32*/, u32 ppn)
{
    memset(dst, 0, PAD_U32 * sizeof(u32));
    memcpy(dst, raw_page(ppn), RAW_U8);
}

/* ================= GPU (CUDA Driver API) ================= */
static CUcontext  g_ctx = NULL;
static CUmodule   g_mod = NULL;
static CUfunction g_k_phase2 = NULL, g_k_phase3 = NULL;

static const char *cu_err(CUresult r)
{
    const char *s = NULL;
    cuGetErrorString(r, &s);
    return s ? s : "unknown";
}
#define CUCHECK(call) do { CUresult _r=(call); if(_r!=CUDA_SUCCESS){ \
    fprintf(stderr,"CUDA error %d (%s) at %s:%d\n",_r,cu_err(_r),__FILE__,__LINE__); return -1; } } while(0)

static char *read_file(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseeko(f, 0, SEEK_END); long n = ftello(f); fseeko(f, 0, SEEK_SET);
    char *b = malloc(n + 1);
    if (!b) { fclose(f); return NULL; }
    if (fread(b, 1, n, f) != (size_t)n) { fclose(f); free(b); return NULL; }
    b[n] = 0; fclose(f);
    if (len) *len = (size_t)n;
    return b;
}

/* find kernel.ptx: explicit path, else next to the executable, else cwd */
static char *find_ptx(const char *explicit_path)
{
    size_t len;
    if (explicit_path) return read_file(explicit_path, &len);
    char exe[4096]; ssize_t r = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (r > 0) {
        exe[r] = 0;
        char *slash = strrchr(exe, '/');
        if (slash) {
            char path[4200];
            snprintf(path, sizeof(path), "%.*s/kernel.ptx", (int)(slash - exe), exe);
            char *b = read_file(path, &len);
            if (b) return b;
        }
    }
    return read_file("kernel.ptx", &len);
}

static int gpu_init(const char *ptx_path)
{
    CUCHECK(cuInit(0));
    int ndev = 0; CUCHECK(cuDeviceGetCount(&ndev));
    if (ndev < 1) { fprintf(stderr, "No CUDA device found\n"); return -1; }
    CUdevice dev; CUCHECK(cuDeviceGet(&dev, 0));
    char name[256] = {0}; cuDeviceGetName(name, sizeof(name), dev);
    printf("GPU: %s\n", name);
    CUCHECK(cuCtxCreate_v2(&g_ctx, 0, dev));

    char *ptx = find_ptx(ptx_path);
    if (!ptx) { fprintf(stderr, "Can't read kernel.ptx (use --ptx <path>)\n"); return -1; }

    char errlog[8192] = {0}; char infolog[2048] = {0};
    CUjit_option opt[4] = {
        CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_INFO_LOG_BUFFER,  CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES };
    void *optv[4] = { errlog, (void *)(size_t)sizeof(errlog),
                      infolog, (void *)(size_t)sizeof(infolog) };
    CUresult lr = cuModuleLoadDataEx(&g_mod, ptx, 4, opt, optv);
    free(ptx);
    if (lr != CUDA_SUCCESS) {
        fprintf(stderr, "PTX JIT failed: %s\n%s\n", cu_err(lr), errlog);
        return -1;
    }
    CUCHECK(cuModuleGetFunction(&g_k_phase2, g_mod, "phase2"));
    CUCHECK(cuModuleGetFunction(&g_k_phase3, g_mod, "phase3"));
    return 0;
}

/* Phase 2 on GPU: returns number of hits, fills seeds[] (up to cap) */
static int gpu_phase2(const u32 *raw0, u32 ppn0, u32 start, u64 total,
                      u32 *seeds, u32 cap)
{
    CUdeviceptr d_raw0, d_seeds, d_count;
    CUCHECK(cuMemAlloc_v2(&d_raw0, PAD_U32 * sizeof(u32)));
    CUCHECK(cuMemAlloc_v2(&d_seeds, cap * sizeof(u32)));
    CUCHECK(cuMemAlloc_v2(&d_count, sizeof(int)));
    CUCHECK(cuMemcpyHtoD_v2(d_raw0, raw0, PAD_U32 * sizeof(u32)));
    CUCHECK(cuMemsetD32_v2(d_count, 0, 1));

    unsigned block = 256, grid = 32768;   /* ~8.4M threads, grid-stride */
    void *args[] = { &d_raw0, &ppn0, &start, &total, &d_seeds, &d_count, &cap };
    CUCHECK(cuLaunchKernel(g_k_phase2, grid, 1, 1, block, 1, 1, 0, 0, args, 0));
    CUCHECK(cuCtxSynchronize());

    int count = 0;
    CUCHECK(cuMemcpyDtoH_v2(&count, d_count, sizeof(int)));
    u32 ncopy = (count < 0) ? 0 : ((u32)count < cap ? (u32)count : cap);
    if (ncopy) CUCHECK(cuMemcpyDtoH_v2(seeds, d_seeds, ncopy * sizeof(u32)));
    cuMemFree_v2(d_raw0); cuMemFree_v2(d_seeds); cuMemFree_v2(d_count);
    return count;
}

/* Phase 3 on GPU: brute high half B against up to two extra flashes */
static int gpu_phase3(u32 seed0, u64 range,
                      const u32 *rawA, u32 ppnA, u32 shiftA, u32 fbA,
                      const u32 *rawB, u32 ppnB, u32 shiftB, u32 fbB,
                      u32 nflash, u32 *bs, u32 cap)
{
    CUdeviceptr d_rawA, d_rawB, d_out, d_count;
    CUCHECK(cuMemAlloc_v2(&d_rawA, PAD_U32 * sizeof(u32)));
    CUCHECK(cuMemAlloc_v2(&d_rawB, PAD_U32 * sizeof(u32)));
    CUCHECK(cuMemAlloc_v2(&d_out, cap * sizeof(u32)));
    CUCHECK(cuMemAlloc_v2(&d_count, sizeof(int)));
    CUCHECK(cuMemcpyHtoD_v2(d_rawA, rawA, PAD_U32 * sizeof(u32)));
    CUCHECK(cuMemcpyHtoD_v2(d_rawB, rawB, PAD_U32 * sizeof(u32)));
    CUCHECK(cuMemsetD32_v2(d_count, 0, 1));

    unsigned block = 256, grid = 8192;
    void *args[] = { &seed0, &range, &d_rawA, &ppnA, &shiftA, &fbA,
                     &d_rawB, &ppnB, &shiftB, &fbB, &nflash,
                     &d_out, &d_count, &cap };
    CUCHECK(cuLaunchKernel(g_k_phase3, grid, 1, 1, block, 1, 1, 0, 0, args, 0));
    CUCHECK(cuCtxSynchronize());

    int count = 0;
    CUCHECK(cuMemcpyDtoH_v2(&count, d_count, sizeof(int)));
    u32 ncopy = (count < 0) ? 0 : ((u32)count < cap ? (u32)count : cap);
    if (ncopy) CUCHECK(cuMemcpyDtoH_v2(bs, d_out, ncopy * sizeof(u32)));
    cuMemFree_v2(d_rawA); cuMemFree_v2(d_rawB); cuMemFree_v2(d_out); cuMemFree_v2(d_count);
    return count;
}

/* ================= CPU fallback (single thread; see memory) ================= */
static int cpu_phase2(const u32 *raw0, u32 ppn0, u32 start, u64 total, u32 *seeds, u32 cap)
{
    int found = 0;
    for (u64 i = 0; i < total; ++i) {
        u32 seed = start + (u32)i;
        if (cpu_sig_ok(raw0, ppn0, seed)) {
            if ((u32)found < cap) seeds[found] = seed;
            found++;
        }
        if ((i & 0xFFFFFFu) == 0)
            fprintf(stderr, "\r  cpu phase2 %.1f%%", 100.0 * (double)i / (double)total);
    }
    fprintf(stderr, "\r                         \r");
    return found;
}

/* ================= main ================= */
static void usage(const char *p)
{
    printf("Usage: %s <nand_dump> [tachyon] [fuseid_hint]\n", p);
    printf("Options:\n");
    printf("  --cpu              brute on CPU (1 thread) instead of GPU\n");
    printf("  --ptx <path>       path to kernel.ptx\n");
    printf("  --start <hex>      phase-2 start seed (default 0)\n");
    printf("  --count <hex>      phase-2 seed count (default 0x100000000)\n");
    printf("  --range3 <hex>     phase-3 high-half search size (default 0x1000000)\n");
    printf("tachyon is metadata only (not used for recovery); default 0x200000.\n");
}

int main(int argc, char **argv)
{
    const char *dump = NULL, *ptx = NULL;
    u32 tachyon = 0x200000u;
    u64 start = 0, count = 0x100000000ULL, range3 = 0x1000000ULL;
    int use_cpu = 0, posarg = 0;

    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--cpu"))    use_cpu = 1;
        else if (!strcmp(argv[i], "--ptx")   && i + 1 < argc) ptx = argv[++i];
        else if (!strcmp(argv[i], "--start") && i + 1 < argc) start = strtoull(argv[++i], 0, 16);
        else if (!strcmp(argv[i], "--count") && i + 1 < argc) count = strtoull(argv[++i], 0, 16);
        else if (!strcmp(argv[i], "--range3")&& i + 1 < argc) range3 = strtoull(argv[++i], 0, 16);
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
        else if (argv[i][0] != '-') {
            if (posarg == 0) dump = argv[i];
            else if (posarg == 1) tachyon = (u32)strtoull(argv[i], 0, 16);
            /* posarg 2 (fuseid hint) accepted and ignored - it is what we recover */
            posarg++;
        }
    }
    if (!dump) { usage(argv[0]); return -1; }
    if (count > 0x100000000ULL) count = 0x100000000ULL;

    if (load_nand(dump)) return -1;
    if (build_lpt())     return -1;

    flash_t fl[16]; int nfl = 0;
    if (walk_flashes(fl, 16, &nfl)) return -1;
    printf("Found %d flash partition(s):\n", nfl);
    for (int i = 0; i < nfl; ++i) {
        int scr = (SCRAMBLE_FLAG >> fl[i].flashnum) & 1;
        printf("  flash%d  PBR lbn=0x%X  ppn=0x%X  scramble_flag=%d\n",
               fl[i].flashnum, fl[i].pbr_lbn, fl[i].ppn, scr);
    }

    /* locate flash0 (must be scrambled) + the extra scrambled flashes */
    flash_t *f0 = NULL, *extra[2] = { NULL, NULL }; int nextra = 0;
    for (int i = 0; i < nfl; ++i) {
        if (fl[i].ppn == (u32)-1) continue;
        if (!((SCRAMBLE_FLAG >> fl[i].flashnum) & 1)) continue;
        if (fl[i].flashnum == 0) f0 = &fl[i];
        else if (nextra < 2) extra[nextra++] = &fl[i];
    }
    if (!f0) { fprintf(stderr, "flash0 PBR not found / not scrambled\n"); return -1; }

    /* if flash0 already reads a valid signature with seed 0, the logic data is
     * not scrambled and the fuse id leaves no footprint in this dump. */
    static u32 pg0[PAD_U32];
    load_padded_page(pg0, f0->ppn);
    if (cpu_sig_ok(pg0, f0->ppn, 0)) {
        printf("\nLogic data is NOT scrambled on this dump - the fuse id is unused\n"
               "here and cannot be recovered from it (needs a FW 3.00+ scrambled NAND).\n");
        return 2;
    }

    printf("\nPhase 2: brute flash0 seed  [start=0x%llX count=0x%llX]  on %s ...\n",
           (unsigned long long)start, (unsigned long long)count, use_cpu ? "CPU" : "GPU");

    if (!use_cpu && gpu_init(ptx)) {
        fprintf(stderr, "GPU init failed; retry with --cpu for the CPU fallback.\n");
        return -1;
    }

    u32 seeds[16]; int nseed;
    if (use_cpu) nseed = cpu_phase2(pg0, f0->ppn, (u32)start, count, seeds, 16);
    else         nseed = gpu_phase2(pg0, f0->ppn, (u32)start, count, seeds, 16);

    if (nseed <= 0) { printf("No flash0 seed found in the searched range.\n"); return 1; }
    u32 seed0 = seeds[0];
    printf("  seed0 = 0x%08X", seed0);
    if (nseed > 1) { printf("  (+%d more:", nseed - 1); for (int i=1;i<nseed && i<16;i++) printf(" 0x%08X", seeds[i]); printf(")"); }
    printf("\n");

    /* Phase 3: resolve the high half B */
    if (nextra == 0) {
        printf("\nOnly flash0 is scrambled - high 32 bits of the fuse id cannot be\n"
               "resolved (need a 2nd scrambled flash). low32 ^ high32 = 0x%08X.\n", seed0);
        return 0;
    }

    static u32 pgA[PAD_U32], pgB[PAD_U32];
    load_padded_page(pgA, extra[0]->ppn);
    u32 shiftA = (u32)((0xFE * extra[0]->flashnum) & 31);
    u32 fbA    = rotr(0xC4536DE6u, -(extra[0]->flashnum));
    u32 shiftB = 0, fbB = 0; u32 nf3 = 1;
    if (nextra >= 2) {
        load_padded_page(pgB, extra[1]->ppn);
        shiftB = (u32)((0xFE * extra[1]->flashnum) & 31);
        fbB    = rotr(0xC4536DE6u, -(extra[1]->flashnum));
        nf3 = 2;
    } else {
        memcpy(pgB, pgA, sizeof(pgA)); /* unused */
    }

    printf("Phase 3: resolve high half B via flash%d%s%d  [range=0x%llX] ...\n",
           extra[0]->flashnum, nextra >= 2 ? "+flash" : "",
           nextra >= 2 ? extra[1]->flashnum : 0, (unsigned long long)range3);

    u32 bs[16]; int nb;
    if (use_cpu) {
        nb = 0;
        for (u64 B = 0; B < range3; ++B) {
            u32 A = seed0 ^ (u32)B;
            u32 sA = A ^ rotr((u32)B, (int)shiftA); if (!sA) sA = fbA;
            if (!cpu_sig_ok(pgA, extra[0]->ppn, sA)) continue;
            if (nf3 >= 2) { u32 sB = A ^ rotr((u32)B,(int)shiftB); if(!sB) sB=fbB;
                            if (!cpu_sig_ok(pgB, extra[1]->ppn, sB)) continue; }
            if ((u32)nb < 16) bs[nb] = (u32)B;
            nb++;
        }
    } else {
        nb = gpu_phase3(seed0, range3,
                        pgA, extra[0]->ppn, shiftA, fbA,
                        pgB, extra[1] ? extra[1]->ppn : 0, shiftB, fbB,
                        nf3, bs, 16);
    }

    if (nb <= 0) {
        printf("No high half found in range 0x%llX (try a larger --range3).\n",
               (unsigned long long)range3);
        return 1;
    }

    /* smallest B is the physically plausible fuse id (high bits are small) */
    u32 bestB = bs[0];
    for (int i = 1; i < nb && i < 16; ++i) if (bs[i] < bestB) bestB = bs[i];
    u64 fuseid = ((u64)bestB << 32) | (u64)(seed0 ^ bestB);

    printf("\n==================================================\n");
    printf(" Recovered Fuse Id : 0x%llX\n", (unsigned long long)fuseid);
    printf(" Tachyon (given)   : 0x%X\n", tachyon);
    printf("   low32  = 0x%08X   high32 = 0x%08X\n", seed0 ^ bestB, bestB);
    printf("   flash0 seed = 0x%08X\n", seed0);
    for (int i = 0; i < nextra; ++i) {
        u32 sh = (u32)((0xFE * extra[i]->flashnum) & 31);
        u32 sd = (seed0 ^ bestB) ^ rotr(bestB, (int)sh);
        if (!sd) sd = rotr(0xC4536DE6u, -(extra[i]->flashnum));
        printf("   flash%d seed = 0x%08X\n", extra[i]->flashnum, sd);
    }
    if (nb > 1) {
        printf(" note: %d high-half candidates; picked smallest. others:", nb);
        for (int i = 0; i < nb && i < 16; ++i) if (bs[i] != bestB) printf(" 0x%X", bs[i]);
        printf("\n");
    }
    printf("==================================================\n");

    if (g_mod) { /* best-effort cleanup */ }
    if (g_ctx) cuCtxDestroy_v2(g_ctx);
    return 0;
}
