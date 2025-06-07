#ifndef TENSOROP_H_   
#define TENSOROP_H_

/* CUTLASS v3.X matrix multiplication and related operations template abstractions library. */
#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"

/* Used for accessing the GPU tensor cores to perform tensorized fused AND+POPC operations. */
cudaError_t tensorPopAnd_batched(int m, int n, int k, int lda, int ldb, int ldc, cudaStream_t stream, void const ** ptr_A_array, void const ** ptr_B_array, void ** ptr_C_array, int numBatches);

#endif
