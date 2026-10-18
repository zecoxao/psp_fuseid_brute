/*
 * cuda_min.h - the slice of the CUDA Driver API this tool uses.
 *
 * The NVIDIA driver (libcuda.so.1) is a stable C ABI, so we can talk to it
 * without the CUDA toolkit installed - we just declare the handful of entry
 * points we need.  The "_v2" suffixes are the real exported symbol names.
 */
#ifndef CUDA_MIN_H
#define CUDA_MIN_H

#include <stddef.h>

typedef int                CUresult;
typedef int                CUdevice;
typedef unsigned long long CUdeviceptr;
typedef struct CUctx_st   *CUcontext;
typedef struct CUmod_st   *CUmodule;
typedef struct CUfunc_st  *CUfunction;
typedef struct CUstream_st*CUstream;
typedef int                CUjit_option;

#define CUDA_SUCCESS 0

#define CU_JIT_INFO_LOG_BUFFER             3
#define CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES  4
#define CU_JIT_ERROR_LOG_BUFFER            5
#define CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES 6

CUresult cuInit(unsigned int);
CUresult cuDeviceGetCount(int *);
CUresult cuDeviceGet(CUdevice *, int);
CUresult cuDeviceGetName(char *, int, CUdevice);
CUresult cuCtxCreate_v2(CUcontext *, unsigned int, CUdevice);
CUresult cuCtxDestroy_v2(CUcontext);
CUresult cuCtxSynchronize(void);
CUresult cuModuleLoadDataEx(CUmodule *, const void *, unsigned int,
                            CUjit_option *, void **);
CUresult cuModuleGetFunction(CUfunction *, CUmodule, const char *);
CUresult cuMemAlloc_v2(CUdeviceptr *, size_t);
CUresult cuMemFree_v2(CUdeviceptr);
CUresult cuMemcpyHtoD_v2(CUdeviceptr, const void *, size_t);
CUresult cuMemcpyDtoH_v2(void *, CUdeviceptr, size_t);
CUresult cuMemsetD32_v2(CUdeviceptr, unsigned int, size_t);
CUresult cuLaunchKernel(CUfunction,
                        unsigned int, unsigned int, unsigned int,
                        unsigned int, unsigned int, unsigned int,
                        unsigned int, CUstream, void **, void **);
CUresult cuGetErrorString(CUresult, const char **);

#endif /* CUDA_MIN_H */
