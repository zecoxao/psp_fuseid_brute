/*
 * psp_nand_fuseid_brute - NVPTX device kernels
 *
 * Compiled to PTX by clang's NVPTX backend (no CUDA toolkit required) and
 * loaded at runtime through the CUDA Driver API (libcuda).  The driver JITs
 * this PTX to whatever NVIDIA GPU is present (Ampere RTX 3050 ... Blackwell
 * RTX 5060 and beyond), so a single build runs everywhere.
 *
 * The kernels reproduce pspNandUnscramblePage()/pspLfatfsGenSeedx1() from the
 * reference extractor bit-for-bit and only inspect the 16-byte PBR signature
 * at offset 0x1F0 of a descrambled page, so they are pure 32-bit integer code
 * (no libdevice dependency).
 */

typedef unsigned int       u32;
typedef unsigned long long u64;

/* ---- NVPTX special registers (clang builtins, return int) ---- */
#define TID    ((u32)__nvvm_read_ptx_sreg_tid_x())
#define NTID   ((u32)__nvvm_read_ptx_sreg_ntid_x())
#define CTAID  ((u32)__nvvm_read_ptx_sreg_ctaid_x())
#define NCTAID ((u32)__nvvm_read_ptx_sreg_nctaid_x())

/* generic-space 32-bit atomic add, returns old value (clang builtin) */

/* The scrambled input page is uploaded padded to 0x184 u32 (0x610 bytes):
 * real page bytes in [0,0x200), zero padding afterwards.  A wrong seed can
 * drive the read offset up to 0x5FF via the &0x5FF mask; padding keeps every
 * access in-bounds and deterministic while leaving the correct-seed result
 * (which only reads [0,0x1FC]) identical to the reference descrambler. */
#define PAGE_U32 0x184u

static inline u32 d_rotr(u32 v, u32 sh)
{
    sh &= 31u;
    if (sh == 0u) return v;
    return (v >> sh) | (v << (32u - sh));
}

static inline u32 d_bitrev(u32 x)
{
    x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
    x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
    x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
    x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
    x = ((x >> 16) & 0xffffu)    | ((x & 0xffffu) << 16);
    return x;
}

/*
 * Descramble one page with the given seed and physical page number, then test
 * the FAT PBR boot signature: 14 zero bytes followed by 0x55 0xAA at 0x1FE.
 * As little-endian u32 at page offset 0x1F0: 0, 0, 0, 0xAA550000.
 */
static inline int d_sig_ok(const u32 *raw, u32 ppn, u32 seed)
{
    u32 s   = d_rotr(seed, 0x15u);
    u32 k   = d_rotr(ppn, 0x11u) ^ ((s << 3) - s);
    u32 sel = 0x10u * ((ppn ^ s) & 0x1Fu);   /* byte offset into the page */
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
            return (e1 == 0u) & (e2 == 0u) & (e3 == 0u) & (e4 == 0xAA550000u);
        sel &= 0x5FFu;
        k = d_bitrev(k + s);
    }
    return 0;
}

/*
 * Phase 2: brute the 32-bit flash0 scramble seed (= low32(fuseid) ^ high32(fuseid)).
 * Grid-stride over [start, start+total).  Hits are appended to out_seeds[] with
 * an atomic counter (first out_cap entries kept).
 */
__attribute__((nvptx_kernel))
void phase2(const u32 *raw0, u32 ppn0, u32 start, u64 total,
            u32 *out_seeds, int *out_count, u32 out_cap)
{
    u64 gid    = (u64)(CTAID * NTID + TID);
    u64 stride = (u64)NCTAID * (u64)NTID;
    for (u64 i = gid; i < total; i += stride) {
        u32 seed = start + (u32)i;
        if (d_sig_ok(raw0, ppn0, seed)) {
            int p = __nvvm_atom_add_gen_i(out_count, 1);
            if ((u32)p < out_cap) out_seeds[p] = seed;
        }
    }
}

/*
 * Phase 3: given the recovered flash0 seed (= A ^ B, A=low32, B=high32), brute
 * the high half B.  For each candidate the per-flash seeds of the other
 * scrambled flashes are reconstructed exactly as pspLfatfsGenSeedx1() does
 * (seed = A ^ ror(B, 0xFE*flashnum), with the ror(0xC4536DE6,-flashnum)
 * fallback when that is zero) and verified against their PBR signatures.
 * A candidate must satisfy every supplied flash, which pins B uniquely.
 */
__attribute__((nvptx_kernel))
void phase3(u32 seed0, u64 range,
            const u32 *rawA, u32 ppnA, u32 shiftA, u32 fbA,
            const u32 *rawB, u32 ppnB, u32 shiftB, u32 fbB,
            u32 nflash,
            u32 *out_B, int *out_count, u32 out_cap)
{
    u64 gid    = (u64)(CTAID * NTID + TID);
    u64 stride = (u64)NCTAID * (u64)NTID;
    for (u64 i = gid; i < range; i += stride) {
        u32 B = (u32)i;
        u32 A = seed0 ^ B;

        u32 sA = A ^ d_rotr(B, shiftA);
        if (sA == 0u) sA = fbA;
        if (!d_sig_ok(rawA, ppnA, sA)) continue;

        if (nflash >= 2u) {
            u32 sB = A ^ d_rotr(B, shiftB);
            if (sB == 0u) sB = fbB;
            if (!d_sig_ok(rawB, ppnB, sB)) continue;
        }

        int p = __nvvm_atom_add_gen_i(out_count, 1);
        if ((u32)p < out_cap) out_B[p] = B;
    }
}
