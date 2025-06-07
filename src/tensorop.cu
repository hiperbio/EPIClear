#include "tensorop.hpp"

#include <iostream>


/* Tested on A100 (Ampere/SM80) but it should also work on newer architectures supported through the CUTLASS 3.X API. */

#define CUDA_ARCH       cutlass::arch::Sm80    
#define TENSOR_OP       cutlass::arch::OpMultiplyAdd
#define MMA_OP_SIZE     cutlass::gemm::GemmShape<16, 8, 256>
#define MMA_WARP_SIZE   cutlass::gemm::GemmShape<64, 64, 512>
#define MMA_TBLOCK_SIZE cutlass::gemm::GemmShape<128, 256, 512>
#define MMA_NUM_STAGES  3


/* Checks for CUTLASS errors and prints error information.
 * from:  'cutlass/examples/common/helper.h' */

#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }


/* Setup of batched AND+POPC 1-bit tensor core accelerated kernel for compatible NVIDIA GPUs. */

// Input matrix A
using         ElementA    = cutlass::uint1b_t;                    		// Data type of matrix (UINT1)
using         LayoutA     = cutlass::layout::RowMajor;                      	// Layout of matrix (Row Major)
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    	// Memory access granularity/alignment

// Input matrix B
using         ElementB    = cutlass::uint1b_t;                                	// Data type of matrix (UINT1)
using         LayoutB     = cutlass::layout::ColumnMajor;                     	// Layout of matrix (Column Major)
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    	// Memory access granularity/alignment

// Output matrix C/D
using         ElementC    = int32_t;                                		// Data type of matrix (INT32)
using         LayoutC     = cutlass::layout::ColumnMajor;                      	// Layout of matrix (Column Major)
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    	// Memory access granularity/alignment

// Multiply-accumulate blocking/pipelining details
using ElementAccumulator  = int32_t;                          			// Data type of accumulation (INT32)
using ArchTag             = CUDA_ARCH;                      			// Compute capability that is being targeted
using OperatorClass       = cutlass::arch::OpClassTensorOp;           		// Operator class (tensor core operation)
using ThreadblockShape    = MMA_TBLOCK_SIZE;   					// Threadblock-level tile size
using WarpShape           = MMA_WARP_SIZE;     					// Warp-level tile size
using InstructionShape    = MMA_OP_SIZE;      					// Instruction-level tile size
constexpr int NumStages   = MMA_NUM_STAGES;                                   	// Global->shared pipeline stages in mainloop


using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
	ElementC,               // Element type of C and D matrices
	AlignmentC,             // Memory access granularity of matrices C and D
	ElementAccumulator,     // Data type of accumulation
	ElementAccumulator>;    // Data type of linear combination


using BinaryMatrixOpKernel = cutlass::gemm::device::GemmUniversal<
ElementA, LayoutA,
	ElementB, LayoutB,
	ElementC, LayoutC,
	ElementAccumulator,
	OperatorClass,
	ArchTag,
	ThreadblockShape,
	WarpShape,
	InstructionShape,
	EpilogueOp,
	cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
	NumStages,
	AlignmentA,
	AlignmentB>;



/* Relies on the CUTLASS 3.X API to perform batched matrix-matrix AND+POPC operations. */
cudaError_t tensorPopAnd_batched(int m, int n, int k, int lda, int ldb, int ldc, cudaStream_t stream, void const ** ptr_A_array, void const ** ptr_B_array, void ** ptr_C_array, int numBatches) {

	typename BinaryMatrixOpKernel::EpilogueOutputOp::Params epilogue_op(1, 0);

	typename BinaryMatrixOpKernel::Arguments kernelArguments(
			cutlass::gemm::GemmUniversalMode::kArray,
			cutlass::gemm::GemmCoord(m,n,k),	// Problem size
			numBatches,	           		// Batch count
			epilogue_op,				// Epilogue parameters (alpha, beta)
			(void const *) ptr_A_array,      	// Pointers to matrices A
			(void const *) ptr_B_array,      	// Pointers to matrices B
			(void const *) ptr_C_array,      	// Pointers to matrices C 
			(void *) ptr_C_array,      		// Pointers to matrices D
			int64_t(),
			int64_t(),
			int64_t(),
			int64_t(),
			int64_t(lda),
			int64_t(ldb),
			int64_t(ldc),
			int64_t(ldc)
			);

	/* Queries and allocates memory for batched matrix-matrix operations */
	size_t workspaceSize = BinaryMatrixOpKernel::get_workspace_size(kernelArguments);
	cutlass::device_memory::allocation<cutlass::uint1b_t> kernelOpWorkspace(workspaceSize);

	/* Instantiates the batched matrix-matrix operations kernel and assesses if problem shape is supported */
	BinaryMatrixOpKernel matrixOpKernel;
	cutlass::Status status = matrixOpKernel.can_implement(kernelArguments);
	CUTLASS_CHECK(status);

	/* Initializes kernel passing the arguments tuple and a pointer to the workspace */
	status = matrixOpKernel.initialize(kernelArguments, kernelOpWorkspace.get(), stream);	
	CUTLASS_CHECK(status);

	/* Launches batched matrix-matrix operations kernel that performs AND+POPC */
	status = matrixOpKernel(stream);
	CUTLASS_CHECK(status);

	return cudaGetLastError();

}



