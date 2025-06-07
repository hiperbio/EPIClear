#include <iostream>
#include <iomanip> 
#include <cfloat>
#include <mpi.h>
#include <omp.h>
#include <cuda.h>
#include <bits/stdc++.h>
#include <cub/cub.cuh>

#include "search.hpp"
#include "helper.hpp"
#include "reduction.hpp"
#include "tensorop.hpp"


#define MAX_BATCH 	4096	

/* Data type of A and B input matrices. */
typedef typename cutlass::Array<cutlass::uint1b_t, 32> ScalarBinary32;


/* Counts the set bits in each bit-pack. */
__global__ void countBits(unsigned int* X, int* counts, int nElems) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < nElems) {
		counts[i] = __popc(X[i]);  
	}
}

/* Compacts {SNP,genotype} bitvectors from a super-block based based on a particular {SNP,genotype} bitvector. */
__global__ void compactBits(const unsigned int* __restrict__ inMat, unsigned int* __restrict__ outMat, const int* __restrict__ pSum, int nElemsOrig, int snpX, int nElemsComp, int firstSnp) {
	__shared__ unsigned int sWriteBuf[TB_SIZE * 2];
	__shared__ int sGlobalWriteStart, sExtraAlign;
	int startSnpIdx = (firstSnp * 2) + (blockDim.x * blockIdx.x + threadIdx.x) * SNPS_PER_THREAD;
	int elemIdx = blockDim.y * blockIdx.y + threadIdx.y;
	if (elemIdx < nElemsOrig) {
		int firstBit = (elemIdx > 0) ? pSum[elemIdx - 1] : 0;
		if(threadIdx.y == 0) {
			int firstBitpack = firstBit / 32;
			sGlobalWriteStart = (((int)(firstBitpack / TB_SIZE))) * TB_SIZE; 
			sExtraAlign = ((firstBitpack & ((TB_SIZE * 2) - 1)) >= TB_SIZE) ? 1 : 0;
		}
		unsigned int xCached = inMat[snpX * nElemsOrig + elemIdx];
		__syncthreads();
		int globalWriteStart = sGlobalWriteStart;
		int firstBitShared = (firstBit - (TB_SIZE * 32 * sExtraAlign)) % ((TB_SIZE * 2) * 32);
		for (int snpIdx = startSnpIdx; snpIdx < (startSnpIdx + SNPS_PER_THREAD); snpIdx += 1) {
			sWriteBuf[threadIdx.y] = 0;
			sWriteBuf[threadIdx.y + TB_SIZE] = 0;
			__syncthreads();
			unsigned int x = xCached;
			unsigned int y = inMat[snpIdx * nElemsOrig + elemIdx];
			int currentBit = firstBitShared;
			while (x != 0) {
				int bitPos = currentBit & 31;
				int sharedIndex = currentBit >> 5;
				unsigned int leastSignificantBit = __ffs(x) - 1;
				unsigned int y_and_one = (y >> leastSignificantBit) & 1u;
				atomicOr(&sWriteBuf[sharedIndex], y_and_one << bitPos);
				currentBit++;
				x &= ~(1u << leastSignificantBit);
			}
			__syncthreads();
			atomicOr(&outMat[snpIdx * nElemsComp + globalWriteStart + threadIdx.y], sWriteBuf[threadIdx.y]);
			atomicOr(&outMat[snpIdx * nElemsComp + globalWriteStart + TB_SIZE + threadIdx.y], sWriteBuf[TB_SIZE + threadIdx.y]);
		}
	}
}


/* Counts the occurences of second-order genotypes resulting from combining a given SNP X in regard to a genotype with a super-block.
   Called after filtering cases and controls in regard to the genotype of a given SNP being processed. */
__global__ void pairPop(int start_SNP_idx, uint *datasetCases, uint *datasetControls, uint *output_individualSNP_popcountsForCases, uint *output_individualSNP_popcountsForControls, int numSNPs, int numCases, int numControls)
{
	/* 'threadIdx.x' is allways 0. It is 'threadIdx.y' that varies between different threads in a thread block */
	uint SNP_i_fromBlockStart = blockDim.x * blockIdx.x + threadIdx.x;	
	uint patient_idx_thread = threadIdx.y;					

	uint SNP_i = start_SNP_idx + SNP_i_fromBlockStart;
	int cases_i, controls_i;

	/* Calculates the number of 32-bit bit-packs to process (">> 5" represents division by 32) */
	int casesSizeNoPadding = (numCases + 31) >> 5;	
	int controlsSizeNoPadding = (numControls + 31) >> 5;

	/* Calculates the number of 32-bit bit-packs to process, including padding */
	int casesSize = ((numCases + PADDING_SAMPLES - 1) / PADDING_SAMPLES) * PADDING_SAMPLES / 32;
	int controlsSize = ((numControls + PADDING_SAMPLES - 1) / PADDING_SAMPLES) * PADDING_SAMPLES / 32;

	int casesZerosAcc = 0;
	int casesOnesAcc = 0;

	int controlsZerosAcc = 0;
	int controlsOnesAcc = 0;

	/* Ensures processing is inside bounds */
	if(SNP_i < numSNPs) {           

		/* Processes cases */
		for(cases_i = 0; cases_i < casesSizeNoPadding; cases_i += blockDim.y) {		
			if((cases_i + patient_idx_thread) < casesSizeNoPadding) {
				casesZerosAcc += __popc(datasetCases[SNP_i_fromBlockStart * SNP_CALC * casesSize + cases_i + patient_idx_thread]);                   
				casesOnesAcc += __popc(datasetCases[SNP_i_fromBlockStart * SNP_CALC * casesSize + casesSize + cases_i + patient_idx_thread]);        
			}
		}

		__syncthreads();

		/* Performs a sum reduction at the level of the thread block. The total is stored in thread 0. */
		int casesZerosAcc_total = blockReduceSum(casesZerosAcc);	
		int casesOnesAcc_total = blockReduceSum(casesOnesAcc);		

		/* Thread 0 writes to global memory */
		if(threadIdx.y == 0) {
			output_individualSNP_popcountsForCases[0 * SUPERBLOCK_SIZE + SNP_i_fromBlockStart] = casesZerosAcc_total;
			output_individualSNP_popcountsForCases[1 * SUPERBLOCK_SIZE + SNP_i_fromBlockStart] = casesOnesAcc_total;
			output_individualSNP_popcountsForCases[2 * SUPERBLOCK_SIZE + SNP_i_fromBlockStart] = numCases - (casesZerosAcc_total + casesOnesAcc_total);
		}

		/* Processes controls */
		for(controls_i = 0; controls_i < controlsSizeNoPadding; controls_i += blockDim.y) {

			if((controls_i + patient_idx_thread) < controlsSizeNoPadding) {
				controlsZerosAcc += __popc(datasetControls[SNP_i_fromBlockStart * SNP_CALC * controlsSize + controls_i + patient_idx_thread]);                   
				controlsOnesAcc += __popc(datasetControls[SNP_i_fromBlockStart * SNP_CALC * controlsSize + controlsSize + controls_i + patient_idx_thread]);        
			}
		}

		__syncthreads();

		/* Performs a sum reduction at the level of the thread block. The total is stored in thread 0. */
		int controlsZerosAcc_total = blockReduceSum(controlsZerosAcc); 
		int controlsOnesAcc_total = blockReduceSum(controlsOnesAcc);   

		if(threadIdx.y == 0) {
			output_individualSNP_popcountsForControls[0 * SUPERBLOCK_SIZE + SNP_i_fromBlockStart] = controlsZerosAcc_total;
			output_individualSNP_popcountsForControls[1 * SUPERBLOCK_SIZE + SNP_i_fromBlockStart] = controlsOnesAcc_total;
			output_individualSNP_popcountsForControls[2 * SUPERBLOCK_SIZE + SNP_i_fromBlockStart] = numControls - (controlsZerosAcc_total + controlsOnesAcc_total);
		}

	}
}

/* Counts the occurences of second-order genotypes resulting from combining two super-blocks. */

__global__ void pairPop_superblocks(uint *datasetCases, uint *datasetControls, uint *output_pairwiseSNP_popcountsForCases, uint *output_pairwiseSNP_popcountsForControls, int numSNPs, int numCases, int numControls, uint SNP_A_start, uint SNP_B_start)
{
	uint SNP_A_i_fromBlockStart = blockDim.x * blockIdx.x + threadIdx.x;
	uint SNP_A_i = SNP_A_start + SNP_A_i_fromBlockStart;

	uint SNP_B_i_fromBlockStart = blockDim.y * blockIdx.y + threadIdx.y;
	uint SNP_B_i = SNP_B_start + SNP_B_i_fromBlockStart;

	int cases_i, controls_i;

	int casesSizeNoPadding = ceil(((float) numCases) / 32.0f);
	int controlsSizeNoPadding = ceil(((float) numControls) / 32.0f);

	int casesSize = ceil(((float) numCases) / ((float) PADDING_SAMPLES)) * PADDING_SAMPLES / 32;
	int controlsSize = ceil(((float) numControls) / ((float) PADDING_SAMPLES)) * PADDING_SAMPLES / 32;

	/* Masks used to take into acount when number of cases or controls is not multiple of 32. */
	uint maskRelevantBitsSetCases = (~0u) << (casesSizeNoPadding * 32 - numCases); 
	uint maskRelevantBitsSetControls = (~0u) << (controlsSizeNoPadding * 32 - numControls); 

	/* Ensures processing is within bounds */
	if((SNP_A_i < numSNPs) && (SNP_B_i < numSNPs)) {       

		int casesCountsArr[9];
		for(int i=0; i<9; i++) {
			casesCountsArr[i] = 0;
		}

		int controlsCountsArr[9];
		for(int i=0; i<9; i++) {
			controlsCountsArr[i] = 0;
		}

		/* Processes cases */

		unsigned int cases_0_A, cases_1_A, cases_2_A, cases_0_B, cases_1_B, cases_2_B;
		for(cases_i = 0; cases_i < (casesSizeNoPadding - 1); cases_i++) {

			cases_0_A = datasetCases[SNP_A_i * SNP_CALC * casesSize + cases_i];
			cases_1_A = datasetCases[SNP_A_i * SNP_CALC * casesSize + casesSize + cases_i];
			cases_2_A = ~(cases_0_A | cases_1_A);

			cases_0_B = datasetCases[SNP_B_i * SNP_CALC * casesSize + cases_i];
			cases_1_B = datasetCases[SNP_B_i * SNP_CALC * casesSize + casesSize + cases_i];
			cases_2_B = ~(cases_0_B | cases_1_B);

			casesCountsArr[0] += __popc(cases_0_A & cases_0_B);
			casesCountsArr[1] += __popc(cases_0_A & cases_1_B);
			casesCountsArr[2] += __popc(cases_0_A & cases_2_B);
			casesCountsArr[3] += __popc(cases_1_A & cases_0_B);
			casesCountsArr[4] += __popc(cases_1_A & cases_1_B);
			casesCountsArr[5] += __popc(cases_1_A & cases_2_B);
			casesCountsArr[6] += __popc(cases_2_A & cases_0_B);
			casesCountsArr[7] += __popc(cases_2_A & cases_1_B);
			casesCountsArr[8] += __popc(cases_2_A & cases_2_B);
		}

		/* Processes last 32-bit bit-pack in order to take into acount when number of cases is not multiple of 32. */

		cases_0_A = datasetCases[SNP_A_i * SNP_CALC * casesSize + cases_i];
		cases_1_A = datasetCases[SNP_A_i * SNP_CALC * casesSize + casesSize + cases_i];
		cases_2_A = (~(cases_0_A | cases_1_A)) & maskRelevantBitsSetCases;

		cases_0_B = datasetCases[SNP_B_i * SNP_CALC * casesSize + cases_i];
		cases_1_B = datasetCases[SNP_B_i * SNP_CALC * casesSize + casesSize + cases_i];
		cases_2_B = (~(cases_0_B | cases_1_B)) & maskRelevantBitsSetCases;

		casesCountsArr[0] += __popc(cases_0_A & cases_0_B);
		casesCountsArr[1] += __popc(cases_0_A & cases_1_B);
		casesCountsArr[2] += __popc(cases_0_A & cases_2_B);
		casesCountsArr[3] += __popc(cases_1_A & cases_0_B);
		casesCountsArr[4] += __popc(cases_1_A & cases_1_B);
		casesCountsArr[5] += __popc(cases_1_A & cases_2_B);
		casesCountsArr[6] += __popc(cases_2_A & cases_0_B);
		casesCountsArr[7] += __popc(cases_2_A & cases_1_B);
		casesCountsArr[8] += __popc(cases_2_A & cases_2_B);

		/* Writes genotype counts to global memory */
		output_pairwiseSNP_popcountsForCases[0LL * (SUPERBLOCK_SIZE * SUPERBLOCK_SIZE) + SNP_B_i_fromBlockStart * SUPERBLOCK_SIZE + SNP_A_i_fromBlockStart] = casesCountsArr[0];
		output_pairwiseSNP_popcountsForCases[1LL * (SUPERBLOCK_SIZE * SUPERBLOCK_SIZE) + SNP_B_i_fromBlockStart * SUPERBLOCK_SIZE + SNP_A_i_fromBlockStart] = casesCountsArr[1];
		output_pairwiseSNP_popcountsForCases[2LL * (SUPERBLOCK_SIZE * SUPERBLOCK_SIZE) + SNP_B_i_fromBlockStart * SUPERBLOCK_SIZE + SNP_A_i_fromBlockStart] = casesCountsArr[2];
		output_pairwiseSNP_popcountsForCases[3LL * (SUPERBLOCK_SIZE * SUPERBLOCK_SIZE) + SNP_B_i_fromBlockStart * SUPERBLOCK_SIZE + SNP_A_i_fromBlockStart] = casesCountsArr[3];
		output_pairwiseSNP_popcountsForCases[4LL * (SUPERBLOCK_SIZE * SUPERBLOCK_SIZE) + SNP_B_i_fromBlockStart * SUPERBLOCK_SIZE + SNP_A_i_fromBlockStart] = casesCountsArr[4];
		output_pairwiseSNP_popcountsForCases[5LL * (SUPERBLOCK_SIZE * SUPERBLOCK_SIZE) + SNP_B_i_fromBlockStart * SUPERBLOCK_SIZE + SNP_A_i_fromBlockStart] = casesCountsArr[5];
		output_pairwiseSNP_popcountsForCases[6LL * (SUPERBLOCK_SIZE * SUPERBLOCK_SIZE) + SNP_B_i_fromBlockStart * SUPERBLOCK_SIZE + SNP_A_i_fromBlockStart] = casesCountsArr[6];
		output_pairwiseSNP_popcountsForCases[7LL * (SUPERBLOCK_SIZE * SUPERBLOCK_SIZE) + SNP_B_i_fromBlockStart * SUPERBLOCK_SIZE + SNP_A_i_fromBlockStart] = casesCountsArr[7];
		output_pairwiseSNP_popcountsForCases[8LL * (SUPERBLOCK_SIZE * SUPERBLOCK_SIZE) + SNP_B_i_fromBlockStart * SUPERBLOCK_SIZE + SNP_A_i_fromBlockStart] = casesCountsArr[8];

		/* Processes contols */

		unsigned int controls_0_A, controls_1_A, controls_2_A, controls_0_B, controls_1_B, controls_2_B;
		for(controls_i = 0; controls_i < (controlsSizeNoPadding - 1); controls_i++) {

			controls_0_A = datasetControls[SNP_A_i * SNP_CALC * controlsSize + controls_i];
			controls_1_A = datasetControls[SNP_A_i * SNP_CALC * controlsSize + controlsSize + controls_i];
			controls_2_A = ~(controls_0_A | controls_1_A);

			controls_0_B = datasetControls[SNP_B_i * SNP_CALC * controlsSize + controls_i];
			controls_1_B = datasetControls[SNP_B_i * SNP_CALC * controlsSize + controlsSize + controls_i];
			controls_2_B = ~(controls_0_B | controls_1_B);

			controlsCountsArr[0] += __popc(controls_0_A & controls_0_B);
			controlsCountsArr[1] += __popc(controls_0_A & controls_1_B);
			controlsCountsArr[2] += __popc(controls_0_A & controls_2_B);
			controlsCountsArr[3] += __popc(controls_1_A & controls_0_B);
			controlsCountsArr[4] += __popc(controls_1_A & controls_1_B);
			controlsCountsArr[5] += __popc(controls_1_A & controls_2_B);
			controlsCountsArr[6] += __popc(controls_2_A & controls_0_B);
			controlsCountsArr[7] += __popc(controls_2_A & controls_1_B);
			controlsCountsArr[8] += __popc(controls_2_A & controls_2_B);
		}

		/* Processes last 32-bit bit-pack in order to take into acount when number of controls is not multiple of 32. */

		controls_0_A = datasetControls[SNP_A_i * SNP_CALC * controlsSize + controls_i];
		controls_1_A = datasetControls[SNP_A_i * SNP_CALC * controlsSize + controlsSize + controls_i];
		controls_2_A = (~(controls_0_A | controls_1_A)) & maskRelevantBitsSetControls;

		controls_0_B = datasetControls[SNP_B_i * SNP_CALC * controlsSize + controls_i];
		controls_1_B = datasetControls[SNP_B_i * SNP_CALC * controlsSize + controlsSize + controls_i];
		controls_2_B = (~(controls_0_B | controls_1_B)) & maskRelevantBitsSetControls;

		controlsCountsArr[0] += __popc(controls_0_A & controls_0_B);
		controlsCountsArr[1] += __popc(controls_0_A & controls_1_B);
		controlsCountsArr[2] += __popc(controls_0_A & controls_2_B);
		controlsCountsArr[3] += __popc(controls_1_A & controls_0_B);
		controlsCountsArr[4] += __popc(controls_1_A & controls_1_B);
		controlsCountsArr[5] += __popc(controls_1_A & controls_2_B);
		controlsCountsArr[6] += __popc(controls_2_A & controls_0_B);
		controlsCountsArr[7] += __popc(controls_2_A & controls_1_B);
		controlsCountsArr[8] += __popc(controls_2_A & controls_2_B);

		/* Writes genotype counts to global memory */
		output_pairwiseSNP_popcountsForControls[0LL * (SUPERBLOCK_SIZE * SUPERBLOCK_SIZE) + SNP_B_i_fromBlockStart * SUPERBLOCK_SIZE + SNP_A_i_fromBlockStart] = controlsCountsArr[0];
		output_pairwiseSNP_popcountsForControls[1LL * (SUPERBLOCK_SIZE * SUPERBLOCK_SIZE) + SNP_B_i_fromBlockStart * SUPERBLOCK_SIZE + SNP_A_i_fromBlockStart] = controlsCountsArr[1];
		output_pairwiseSNP_popcountsForControls[2LL * (SUPERBLOCK_SIZE * SUPERBLOCK_SIZE) + SNP_B_i_fromBlockStart * SUPERBLOCK_SIZE + SNP_A_i_fromBlockStart] = controlsCountsArr[2];
		output_pairwiseSNP_popcountsForControls[3LL * (SUPERBLOCK_SIZE * SUPERBLOCK_SIZE) + SNP_B_i_fromBlockStart * SUPERBLOCK_SIZE + SNP_A_i_fromBlockStart] = controlsCountsArr[3];
		output_pairwiseSNP_popcountsForControls[4LL * (SUPERBLOCK_SIZE * SUPERBLOCK_SIZE) + SNP_B_i_fromBlockStart * SUPERBLOCK_SIZE + SNP_A_i_fromBlockStart] = controlsCountsArr[4];
		output_pairwiseSNP_popcountsForControls[5LL * (SUPERBLOCK_SIZE * SUPERBLOCK_SIZE) + SNP_B_i_fromBlockStart * SUPERBLOCK_SIZE + SNP_A_i_fromBlockStart] = controlsCountsArr[5];
		output_pairwiseSNP_popcountsForControls[6LL * (SUPERBLOCK_SIZE * SUPERBLOCK_SIZE) + SNP_B_i_fromBlockStart * SUPERBLOCK_SIZE + SNP_A_i_fromBlockStart] = controlsCountsArr[6];
		output_pairwiseSNP_popcountsForControls[7LL * (SUPERBLOCK_SIZE * SUPERBLOCK_SIZE) + SNP_B_i_fromBlockStart * SUPERBLOCK_SIZE + SNP_A_i_fromBlockStart] = controlsCountsArr[7];
		output_pairwiseSNP_popcountsForControls[8LL * (SUPERBLOCK_SIZE * SUPERBLOCK_SIZE) + SNP_B_i_fromBlockStart * SUPERBLOCK_SIZE + SNP_A_i_fromBlockStart] = controlsCountsArr[8];

	}
}


/* Construction of 2 x 8 contingency table values from the output of the GPU tensor cores.
 * Derivation of remaining 2 x 19 contingency table values.
 * Scoring of sets of SNPs and idenfication of best score (and corresponding set).
 */
template <bool doCheck> __global__ void objectiveFunctionKernel(int *C_ptrGPU_cases, int *C_ptrGPU_controls, uint * d_output_individualSNP_popcountsForCases_filteredBy_X0_super1, uint * d_output_individualSNP_popcountsForControls_filteredBy_X0_super1, uint * d_output_individualSNP_popcountsForCases_filteredBy_X1_super1, uint * d_output_individualSNP_popcountsForControls_filteredBy_X1_super1, uint * d_output_individualSNP_popcountsForCases_filteredBy_X0_super2, uint * d_output_individualSNP_popcountsForControls_filteredBy_X0_super2, uint * d_output_individualSNP_popcountsForCases_filteredBy_X1_super2, uint * d_output_individualSNP_popcountsForControls_filteredBy_X1_super2, float *tablePrecalc, float *output, unsigned long long int *output_packedIndices, int start_Y, int start_Z, int snp_X_index, uint * d_output_pairwiseSNP_popcountsForCases, uint * d_output_pairwiseSNP_popcountsForControls, int numSNPs, int numCases, int numControls)  
{
	int SNP_Y = blockDim.y * blockIdx.y + threadIdx.y;

	/* Stores the index of the SNP (Z) that is part of the SNP triplet with the minimum score. */
	int SNP_Z_withBestScore;		

	float score = FLT_MAX;

	unsigned int superblock_Y_start = floor((double)(start_Y + SNP_Y)  / (double)SUPERBLOCK_SIZE) * SUPERBLOCK_SIZE;


	/* Ensures processing is within bounds. */
	if( ((start_Y + SNP_Y) < numSNPs)) {	

		for(int Z_i = 0; Z_i < BLOCK_OBJFUN; Z_i++) {

			int SNP_Z_index = (blockDim.x * blockIdx.x + threadIdx.x) * BLOCK_OBJFUN + Z_i;	// SNP_Z	
			unsigned int superblock_Z_start = floor((double)(start_Z + SNP_Z_index) / (double)SUPERBLOCK_SIZE) * SUPERBLOCK_SIZE;

			/* Checks if computation is within bounds (only when dealing with the last block, in order to avoid significant overhead). */
			if(doCheck == true) {
				if((start_Z + SNP_Z_index) >= numSNPs) {
					break;
				}
			}

			unsigned int SNP_Y_fromSuperChunkStart = (start_Y + SNP_Y) - superblock_Y_start;
			unsigned int SNP_Z_fromSuperChunkStart = (start_Z + SNP_Z_index) - superblock_Z_start;


			/* Stores the 8 genotype counts determined processing the dataset: {0,0,0}, {0,0,1}, {0,1,0}, {0,1,1}, {1,0,0}, {1,0,1}, {1,1,0}, {1,1,1} */
			int calc_arr_cases[8];
			int calc_arr_controls[8];

			/* Stores the score */
			float score_new = 0;


			/* Temporary variables for counts that include X0 */
			int cases_val_x0 = 0;
			int controls_val_x0 = 0;
			int cases_val_extra_x0 = 0;
			int controls_val_extra_x0 = 0;

			/* Temporary variables for counts that include X1 */
			int cases_val_x1 = 0;
			int controls_val_x1 = 0;
			int cases_val_extra_x1 = 0;
			int controls_val_extra_x1 = 0;

			/* Temporary variables for counts that include X2 */
			int cases_val_x2 = 0;
			int controls_val_x2 = 0;
			int cases_val_extra_x2 = 0;
			int controls_val_extra_x2 = 0;

			for(int j = 0; j<SNP_CALC; j++) {
				for(int k = 0; k<SNP_CALC; k++) {

					/* Calculates score contribution from genotypes with X0, i.e. {0,0,0}, {0,0,1}, {0,1,0}, {0,1,1} */
					calc_arr_cases[0*4 + j*2 + k] = C_ptrGPU_cases[0 * ((BLOCK_SIZE * SNP_CALC) * (BLOCK_SIZE * SNP_CALC)) + ((SNP_Z_index * SNP_CALC) * (BLOCK_SIZE * SNP_CALC)) + (k * (BLOCK_SIZE * SNP_CALC)) + (SNP_Y * SNP_CALC) + j];
					calc_arr_controls[0*4 + j*2 + k] = C_ptrGPU_controls[0 * ((BLOCK_SIZE * SNP_CALC) * (BLOCK_SIZE * SNP_CALC)) + ((SNP_Z_index * SNP_CALC) * (BLOCK_SIZE * SNP_CALC)) + (k * (BLOCK_SIZE * SNP_CALC)) + (SNP_Y * SNP_CALC) + j];

					score_new += __ldg(&tablePrecalc[calc_arr_controls[0*4 + j*2 + k]]) + __ldg(&tablePrecalc[calc_arr_cases[0*4 + j*2 + k]]) - __ldg(&tablePrecalc[calc_arr_controls[0*4 + j*2 + k] + calc_arr_cases[0*4 + j*2 + k] + 1]); 

					/* Calculates score contribution from genotypes with X1, i.e. {1,0,0}, {1,0,1}, {1,1,0}, {1,1,1} */
					calc_arr_cases[1*4 + j*2 + k] = C_ptrGPU_cases[1 * ((BLOCK_SIZE * SNP_CALC) * (BLOCK_SIZE * SNP_CALC)) + ((SNP_Z_index * SNP_CALC) * (BLOCK_SIZE * SNP_CALC)) + (k * (BLOCK_SIZE * SNP_CALC)) + (SNP_Y * SNP_CALC) + j];
					calc_arr_controls[1*4 + j*2 + k] = C_ptrGPU_controls[1 * ((BLOCK_SIZE * SNP_CALC) * (BLOCK_SIZE * SNP_CALC)) + ((SNP_Z_index * SNP_CALC) * (BLOCK_SIZE * SNP_CALC)) + (k * (BLOCK_SIZE * SNP_CALC)) + (SNP_Y * SNP_CALC) + j];

					score_new += __ldg(&tablePrecalc[calc_arr_controls[1*4 + j*2 + k]]) + __ldg(&tablePrecalc[calc_arr_cases[1*4 + j*2 + k]]) - __ldg(&tablePrecalc[calc_arr_controls[1*4 + j*2 + k] + calc_arr_cases[1*4 + j*2 + k] + 1]);

					/* Calculates score contribution from genotypes with X2 (infered), i.e. {2,0,0}, {2,0,1}, {2,1,0}, {2,1,1} */
					cases_val_x1 = d_output_pairwiseSNP_popcountsForCases[(j*3 + k) * (SUPERBLOCK_SIZE * SUPERBLOCK_SIZE) + ( SNP_Z_fromSuperChunkStart ) * SUPERBLOCK_SIZE + SNP_Y_fromSuperChunkStart] - (calc_arr_cases[0*4 + j*2 + k] + calc_arr_cases[1*4 + j*2 + k]);
					controls_val_x1 = d_output_pairwiseSNP_popcountsForControls[(j*3 + k) * (SUPERBLOCK_SIZE * SUPERBLOCK_SIZE) + ( SNP_Z_fromSuperChunkStart ) * SUPERBLOCK_SIZE + SNP_Y_fromSuperChunkStart] - (calc_arr_controls[0*4 + j*2 + k] + calc_arr_controls[1*4 + j*2 + k]);
					score_new += __ldg(&tablePrecalc[controls_val_x1]) + __ldg(&tablePrecalc[cases_val_x1]) - __ldg(&tablePrecalc[controls_val_x1 + cases_val_x1 + 1]);

				}
			}


			/* {0,0,2} = {0,0,:} - ({0,0,0} + {0,0,1}) */
			cases_val_x0 = d_output_individualSNP_popcountsForCases_filteredBy_X0_super1[0 * SUPERBLOCK_SIZE + SNP_Y_fromSuperChunkStart] - (calc_arr_cases[0*4 + 0*2 + 0] + calc_arr_cases[0*4 + 0*2 + 1]);
			controls_val_x0 = d_output_individualSNP_popcountsForControls_filteredBy_X0_super1[0 * SUPERBLOCK_SIZE + SNP_Y_fromSuperChunkStart] - (calc_arr_controls[0*4 + 0*2 + 0] + calc_arr_controls[0*4 + 0*2 + 1]);
			score_new += (__ldg(&tablePrecalc[controls_val_x0]) + __ldg(&tablePrecalc[cases_val_x0]) - __ldg(&tablePrecalc[cases_val_x0 + controls_val_x0 + 1]));

			/* {1,0,2} = {1,0,:} - ({1,0,0} + {1,0,1}) */	
			cases_val_x1 = d_output_individualSNP_popcountsForCases_filteredBy_X1_super1[0 * SUPERBLOCK_SIZE + SNP_Y_fromSuperChunkStart] - (calc_arr_cases[1*4 + 0*2 + 0] + calc_arr_cases[1*4 + 0*2 + 1]);
			controls_val_x1 = d_output_individualSNP_popcountsForControls_filteredBy_X1_super1[0 * SUPERBLOCK_SIZE + SNP_Y_fromSuperChunkStart] - (calc_arr_controls[1*4 + 0*2 + 0] + calc_arr_controls[1*4 + 0*2 + 1]);
			score_new += (__ldg(&tablePrecalc[controls_val_x1]) + __ldg(&tablePrecalc[cases_val_x1]) - __ldg(&tablePrecalc[cases_val_x1 + controls_val_x1 + 1]));

			/* {2,0,2} = {:,0,2} - ({0,0,2} + {1,0,2}) */
			cases_val_x1 = d_output_pairwiseSNP_popcountsForCases[(0*3 + 2) * (SUPERBLOCK_SIZE * SUPERBLOCK_SIZE) + ( SNP_Z_fromSuperChunkStart ) * SUPERBLOCK_SIZE + SNP_Y_fromSuperChunkStart] - (cases_val_x0 + cases_val_x1);
			controls_val_x1 = d_output_pairwiseSNP_popcountsForControls[(0*3 + 2) * (SUPERBLOCK_SIZE * SUPERBLOCK_SIZE) + ( SNP_Z_fromSuperChunkStart ) * SUPERBLOCK_SIZE + SNP_Y_fromSuperChunkStart] - (controls_val_x0 + controls_val_x1);
			score_new += (__ldg(&tablePrecalc[controls_val_x1]) + __ldg(&tablePrecalc[cases_val_x1]) - __ldg(&tablePrecalc[cases_val_x1 + controls_val_x1 + 1]));


			/* {0,1,2} = {0,1,:} - ({0,1,0} + {0,1,1}) */
			cases_val_x0 = d_output_individualSNP_popcountsForCases_filteredBy_X0_super1[1 * SUPERBLOCK_SIZE + SNP_Y_fromSuperChunkStart] - (calc_arr_cases[0*4 + 1*2 + 0] + calc_arr_cases[0*4 + 1*2 + 1]);
			controls_val_x0 = d_output_individualSNP_popcountsForControls_filteredBy_X0_super1[1 * SUPERBLOCK_SIZE + SNP_Y_fromSuperChunkStart] - (calc_arr_controls[0*4 + 1*2 + 0] + calc_arr_controls[0*4 + 1*2 + 1]);
			score_new += (__ldg(&tablePrecalc[controls_val_x0]) + __ldg(&tablePrecalc[cases_val_x0]) - __ldg(&tablePrecalc[cases_val_x0 + controls_val_x0 + 1]));

			/* {1,1,2} = {1,1,:} - ({1,1,0} + {1,1,1}) */
			cases_val_x1 = d_output_individualSNP_popcountsForCases_filteredBy_X1_super1[1 * SUPERBLOCK_SIZE + SNP_Y_fromSuperChunkStart] - (calc_arr_cases[1*4 + 1*2 + 0] + calc_arr_cases[1*4 + 1*2 + 1]);
			controls_val_x1 = d_output_individualSNP_popcountsForControls_filteredBy_X1_super1[1 * SUPERBLOCK_SIZE + SNP_Y_fromSuperChunkStart] - (calc_arr_controls[1*4 + 1*2 + 0] + calc_arr_controls[1*4 + 1*2 + 1]);
			score_new += (__ldg(&tablePrecalc[controls_val_x1]) + __ldg(&tablePrecalc[cases_val_x1]) - __ldg(&tablePrecalc[cases_val_x1 + controls_val_x1 + 1]));

			/* {2,1,2} = {:,1,2} - ({0,1,2} + {1,1,2}) */
			cases_val_x1 = d_output_pairwiseSNP_popcountsForCases[(1*3 + 2) * (SUPERBLOCK_SIZE * SUPERBLOCK_SIZE) + ( SNP_Z_fromSuperChunkStart ) * SUPERBLOCK_SIZE + SNP_Y_fromSuperChunkStart] - (cases_val_x0 + cases_val_x1);
			controls_val_x1 = d_output_pairwiseSNP_popcountsForControls[(1*3 + 2) * (SUPERBLOCK_SIZE * SUPERBLOCK_SIZE) + ( SNP_Z_fromSuperChunkStart ) * SUPERBLOCK_SIZE + SNP_Y_fromSuperChunkStart] - (controls_val_x0 + controls_val_x1);
			score_new += (__ldg(&tablePrecalc[controls_val_x1]) + __ldg(&tablePrecalc[cases_val_x1]) - __ldg(&tablePrecalc[cases_val_x1 + controls_val_x1 + 1]));


			/* {0,2,0} = {0,:,0} - ({0,0,0} + {0,1,0}) */
			cases_val_x0 = d_output_individualSNP_popcountsForCases_filteredBy_X0_super2[0 * SUPERBLOCK_SIZE + SNP_Z_fromSuperChunkStart] - (calc_arr_cases[0*4 + 0*2 + 0] + calc_arr_cases[0*4 + 1*2 + 0]);
			controls_val_x0 = d_output_individualSNP_popcountsForControls_filteredBy_X0_super2[0 * SUPERBLOCK_SIZE + SNP_Z_fromSuperChunkStart] - (calc_arr_controls[0*4 + 0*2 + 0] + calc_arr_controls[0*4 + 1*2 + 0]);
			score_new += (__ldg(&tablePrecalc[controls_val_x0]) + __ldg(&tablePrecalc[cases_val_x0]) - __ldg(&tablePrecalc[cases_val_x0 + controls_val_x0 + 1]));

			/* {1,2,0} = {1,:,0} - ({1,0,0} + {1,1,0}) */
			cases_val_x1 = d_output_individualSNP_popcountsForCases_filteredBy_X1_super2[0 * SUPERBLOCK_SIZE + SNP_Z_fromSuperChunkStart] - (calc_arr_cases[1*4 + 0*2 + 0] + calc_arr_cases[1*4 + 1*2 + 0]);
			controls_val_x1 = d_output_individualSNP_popcountsForControls_filteredBy_X1_super2[0 * SUPERBLOCK_SIZE + SNP_Z_fromSuperChunkStart] - (calc_arr_controls[1*4 + 0*2 + 0] + calc_arr_controls[1*4 + 1*2 + 0]);
			score_new += (__ldg(&tablePrecalc[controls_val_x1]) + __ldg(&tablePrecalc[cases_val_x1]) - __ldg(&tablePrecalc[cases_val_x1 + controls_val_x1 + 1]));

			/* {2,2,0} = {:,2,0} - ({0,2,0} + {1,2,0}) */
			cases_val_x2 = d_output_pairwiseSNP_popcountsForCases[(2*3 + 0) * (SUPERBLOCK_SIZE * SUPERBLOCK_SIZE) + ( SNP_Z_fromSuperChunkStart ) * SUPERBLOCK_SIZE + SNP_Y_fromSuperChunkStart] - (cases_val_x0 + cases_val_x1);
			controls_val_x2= d_output_pairwiseSNP_popcountsForControls[(2*3 + 0) * (SUPERBLOCK_SIZE * SUPERBLOCK_SIZE) + ( SNP_Z_fromSuperChunkStart ) * SUPERBLOCK_SIZE + SNP_Y_fromSuperChunkStart] - (controls_val_x0 + controls_val_x1);
			score_new += (__ldg(&tablePrecalc[controls_val_x2]) + __ldg(&tablePrecalc[cases_val_x2]) - __ldg(&tablePrecalc[cases_val_x2 + controls_val_x2 + 1]));


			/* {0,2,1} = {0,:,1} - ({0,0,1} + {0,1,1})
			   Counts stored in extra variables, which are used later to infer {2,2,1} and {0,2,2} */
			cases_val_extra_x0 = d_output_individualSNP_popcountsForCases_filteredBy_X0_super2[1 * SUPERBLOCK_SIZE + SNP_Z_fromSuperChunkStart] - (calc_arr_cases[0*4 + 0*2 + 1] + calc_arr_cases[0*4 + 1*2 + 1]);
			controls_val_extra_x0 = d_output_individualSNP_popcountsForControls_filteredBy_X0_super2[1 * SUPERBLOCK_SIZE + SNP_Z_fromSuperChunkStart] - (calc_arr_controls[0*4 + 0*2 + 1] + calc_arr_controls[0*4 + 1*2 + 1]);
			score_new += (__ldg(&tablePrecalc[controls_val_extra_x0]) + __ldg(&tablePrecalc[cases_val_extra_x0]) - __ldg(&tablePrecalc[cases_val_extra_x0 + controls_val_extra_x0 + 1]));

			/* {1,2,1} = {1,:,1} - ({1,0,1} + {1,1,1})
			   Counts stored in extra variables, which are used later to infer {2,2,1} and {1,2,2} */
			cases_val_extra_x1 = d_output_individualSNP_popcountsForCases_filteredBy_X1_super2[1 * SUPERBLOCK_SIZE + SNP_Z_fromSuperChunkStart] - (calc_arr_cases[1*4 + 0*2 + 1] + calc_arr_cases[1*4 + 1*2 + 1]);
			controls_val_extra_x1 = d_output_individualSNP_popcountsForControls_filteredBy_X1_super2[1 * SUPERBLOCK_SIZE + SNP_Z_fromSuperChunkStart] - (calc_arr_controls[1*4 + 0*2 + 1] + calc_arr_controls[1*4 + 1*2 + 1]);
			score_new += (__ldg(&tablePrecalc[controls_val_extra_x1]) + __ldg(&tablePrecalc[cases_val_extra_x1]) - __ldg(&tablePrecalc[cases_val_extra_x1 + controls_val_extra_x1 + 1]));

			/* {2,2,1} = {:,2,1} - ({0,2,1} + {1,2,1}) */
			cases_val_extra_x2 = d_output_pairwiseSNP_popcountsForCases[(2*3 + 1) * (SUPERBLOCK_SIZE * SUPERBLOCK_SIZE) + ( SNP_Z_fromSuperChunkStart ) * SUPERBLOCK_SIZE + SNP_Y_fromSuperChunkStart] - (cases_val_extra_x0 + cases_val_extra_x1);
			controls_val_extra_x2= d_output_pairwiseSNP_popcountsForControls[(2*3 + 1) * (SUPERBLOCK_SIZE * SUPERBLOCK_SIZE) + ( SNP_Z_fromSuperChunkStart ) * SUPERBLOCK_SIZE + SNP_Y_fromSuperChunkStart] - (controls_val_extra_x0 + controls_val_extra_x1);
			score_new += (__ldg(&tablePrecalc[controls_val_extra_x2]) + __ldg(&tablePrecalc[cases_val_extra_x2]) - __ldg(&tablePrecalc[cases_val_extra_x2 + controls_val_extra_x2 + 1]));


			/* {0,2,2} = {0,2,:} - ({0,2,0} + {0,2,1}) */  
			cases_val_x0 = d_output_individualSNP_popcountsForCases_filteredBy_X0_super1[2 * SUPERBLOCK_SIZE + SNP_Y_fromSuperChunkStart] - (cases_val_x0 + cases_val_extra_x0);
			controls_val_x0 = d_output_individualSNP_popcountsForControls_filteredBy_X0_super1[2 * SUPERBLOCK_SIZE + SNP_Y_fromSuperChunkStart] - (controls_val_x0 + controls_val_extra_x0);
			score_new += (__ldg(&tablePrecalc[controls_val_x0]) + __ldg(&tablePrecalc[cases_val_x0]) - __ldg(&tablePrecalc[cases_val_x0 + controls_val_x0 + 1]));

			/* {1,2,2} = {1,2,:} - ({1,2,0} + {1,2,1}) */
			cases_val_x1 = d_output_individualSNP_popcountsForCases_filteredBy_X1_super1[2 * SUPERBLOCK_SIZE + SNP_Y_fromSuperChunkStart] - (cases_val_x1 + cases_val_extra_x1);
			controls_val_x1 = d_output_individualSNP_popcountsForControls_filteredBy_X1_super1[2 * SUPERBLOCK_SIZE + SNP_Y_fromSuperChunkStart] - (controls_val_x1 + controls_val_extra_x1);
			score_new += (__ldg(&tablePrecalc[controls_val_x1]) + __ldg(&tablePrecalc[cases_val_x1]) - __ldg(&tablePrecalc[cases_val_x1 + controls_val_x1 + 1]));

			/* {2,2,2} = {:,2,2} - ({0,2,2} + {1,2,2}) */
			cases_val_x1 = d_output_pairwiseSNP_popcountsForCases[((long long) (2*3 + 2)) * (SUPERBLOCK_SIZE * SUPERBLOCK_SIZE) + ( SNP_Z_fromSuperChunkStart ) * SUPERBLOCK_SIZE + SNP_Y_fromSuperChunkStart] - (cases_val_x0 + cases_val_x1);
			controls_val_x1= d_output_pairwiseSNP_popcountsForControls[((long long) (2*3 + 2)) * (SUPERBLOCK_SIZE * SUPERBLOCK_SIZE) + ( SNP_Z_fromSuperChunkStart ) * SUPERBLOCK_SIZE + SNP_Y_fromSuperChunkStart] - (controls_val_x0 + controls_val_x1);
			score_new += (__ldg(&tablePrecalc[controls_val_x1]) + __ldg(&tablePrecalc[cases_val_x1]) - __ldg(&tablePrecalc[cases_val_x1 + controls_val_x1 + 1]));


			score_new = fabs(score_new);


			if ((score_new < score) && (snp_X_index < (start_Y + SNP_Y)) && ((start_Y + SNP_Y) < (start_Z + SNP_Z_index))) {	
				SNP_Z_withBestScore = start_Z + SNP_Z_index;
				score = score_new;

			}

		}
	}


	float min_score =  blockReduceMin(score);
	if(threadIdx.x == 0 && threadIdx.y == 0) {
		atomicMin_g_f(output, min_score);
	}

	if(score == min_score) {
		if((snp_X_index < (start_Y + SNP_Y)) &&  ((start_Y + SNP_Y) < SNP_Z_withBestScore)) {
			unsigned long long int packedIndices = (((unsigned long long int)snp_X_index) << 0) | (((unsigned long long int)(start_Y + SNP_Y)) << 21) | (((unsigned long long int)SNP_Z_withBestScore) << 42);
			atomicMinGetIndex(output, min_score, output_packedIndices, packedIndices);
		}
	}

}


template __global__ void objectiveFunctionKernel<true>(int *C_ptrGPU_cases, int *C_ptrGPU_controls, uint * d_output_individualSNP_popcountsForCases_filteredBy_X0_super1, uint * d_output_individualSNP_popcountsForControls_filteredBy_X0_super1, uint * d_output_individualSNP_popcountsForCases_filteredBy_X1_super1, uint * d_output_individualSNP_popcountsForControls_filteredBy_X1_super1, uint * d_output_individualSNP_popcountsForCases_filteredBy_X0_super2, uint * d_output_individualSNP_popcountsForControls_filteredBy_X0_super2, uint * d_output_individualSNP_popcountsForCases_filteredBy_X1_super2, uint * d_output_individualSNP_popcountsForControls_filteredBy_X1_super2, float *tablePrecalc, float *output, unsigned long long int *output_packedIndices, int start_Y, int start_Z, int snp_X_index, uint * d_output_pairwiseSNP_popcountsForCases, uint * d_output_pairwiseSNP_popcountsForControls, int numSNPs, int numCases, int numControls);	

template __global__ void objectiveFunctionKernel<false>(int *C_ptrGPU_cases, int *C_ptrGPU_controls, uint * d_output_individualSNP_popcountsForCases_filteredBy_X0_super1, uint * d_output_individualSNP_popcountsForControls_filteredBy_X0_super1, uint * d_output_individualSNP_popcountsForCases_filteredBy_X1_super1, uint * d_output_individualSNP_popcountsForControls_filteredBy_X1_super1, uint * d_output_individualSNP_popcountsForCases_filteredBy_X0_super2, uint * d_output_individualSNP_popcountsForControls_filteredBy_X0_super2, uint * d_output_individualSNP_popcountsForCases_filteredBy_X1_super2, uint * d_output_individualSNP_popcountsForControls_filteredBy_X1_super2, float *tablePrecalc, float *output, unsigned long long int *output_packedIndices, int start_Y, int start_Z, int snp_X_index, uint * d_output_pairwiseSNP_popcountsForCases, uint * d_output_pairwiseSNP_popcountsForControls, int numSNPs, int numCases, int numControls);	


/* Search function executed by each MPI worker. */
cudaError_t EpistasisDetectionSearch(unsigned int* datasetCases_host_matrixA, unsigned int* datasetControls_host_matrixA, int numSNPs, int numCases, int numControls, uint numSNPsWithPadding, int numCasesWithPadding, int numControlsWithPadding, float * outputFromGpu, unsigned long long int * output_indexFromGpu_packedIndices, int mpiRank) {
	cudaError_t result;

	int *d_counts_X0_cases, *d_counts_X1_cases, *d_counts_X0_controls, *d_counts_X1_controls;
	int *d_prefixSum_X0_cases, *d_prefixSum_X1_cases, *d_prefixSum_X0_controls, *d_prefixSum_X1_controls;

	/* Allocates GPU memory for prefix sum arrays pertaining to cases */
	cudaMalloc(&d_counts_X0_cases, (numCases / 32) * sizeof(unsigned int));
	cudaMalloc(&d_counts_X1_cases, (numCases / 32) * sizeof(unsigned int));
	cudaMalloc(&d_prefixSum_X0_cases, (numCases / 32) * sizeof(unsigned int));
	cudaMalloc(&d_prefixSum_X1_cases, (numCases / 32) * sizeof(unsigned int));

	/* Allocates GPU memory for prefix sum arrays pertaining to controls */
	cudaMalloc(&d_counts_X0_controls, (numControls / 32) * sizeof(unsigned int));
	cudaMalloc(&d_counts_X1_controls, (numControls / 32) * sizeof(unsigned int));
	cudaMalloc(&d_prefixSum_X0_controls, (numControls / 32) * sizeof(unsigned int));
	cudaMalloc(&d_prefixSum_X1_controls, (numControls / 32) * sizeof(unsigned int));

	/* Allocates temporary storage on the GPU for prefix sum operation pertaining to cases */
	void *d_tempStorage_x0_cases = NULL;
	size_t tempStorageSize_x0_cases = 0;
	void *d_tempStorage_x1_cases = NULL;
	size_t tempStorageSize_x1_cases = 0;

	/* Allocates temporary storage on the GPU for prefix sum operation pertaining to controls */
	void *d_tempStorage_x0_controls = NULL;
	size_t tempStorageSize_x0_controls = 0;
	void *d_tempStorage_x1_controls = NULL;
	size_t tempStorageSize_x1_controls = 0;


	/* Allocates memory on the Host for arrays of pointers to A, B and C matrices pertaining to cases */
	void const ** host_A_array_cases = (void const **) malloc(SNP_CALC * MAX_BATCH * sizeof(void const *)); 
	void const ** host_B_array_cases = (void const **) malloc(SNP_CALC * MAX_BATCH * sizeof(void const *)); 
	void ** host_C_array_cases = (void **) malloc(SNP_CALC * MAX_BATCH * sizeof(void *));

	/* Allocates memory on the GPU for arrays of pointers to A, B and C matrices pertaining to cases */

	void const ** ptr_A_array_cases;
	result = cudaMalloc((void const **)&ptr_A_array_cases, SNP_CALC * MAX_BATCH * sizeof(void const *)); 
	if (result != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(result));
	} 

	void const ** ptr_B_array_cases;
	result = cudaMalloc((void const **)&ptr_B_array_cases, SNP_CALC * MAX_BATCH * sizeof(void const *));
	if (result != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(result));
	}

	void ** ptr_C_array_cases;
	result = cudaMalloc((void **)&ptr_C_array_cases, SNP_CALC * MAX_BATCH * sizeof(void *));
	if (result != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(result));
	}


	/* Allocates memory on the Host for arrays of pointers to A, B and C matrices pertaining to controls */

	void const ** host_A_array_controls = (void const **) malloc(SNP_CALC * MAX_BATCH * sizeof(void const *));
	void const ** host_B_array_controls = (void const **) malloc(SNP_CALC * MAX_BATCH * sizeof(void const *));
	void ** host_C_array_controls = (void **) malloc(SNP_CALC * MAX_BATCH * sizeof(void *));


	/* Allocates memory on the GPU for arrays of pointers to A, B and C matrices pertaining to controls */

	void const ** ptr_A_array_controls;
	result = cudaMalloc((void const **)&ptr_A_array_controls, SNP_CALC * MAX_BATCH * sizeof(void const *));
	if (result != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(result));
	}

	void const ** ptr_B_array_controls;
	result = cudaMalloc((void const **)&ptr_B_array_controls, SNP_CALC * MAX_BATCH * sizeof(void const *));
	if (result != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(result));
	}

	void ** ptr_C_array_controls;
	result = cudaMalloc((void **)&ptr_C_array_controls, SNP_CALC * MAX_BATCH * sizeof(void *));
	if (result != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(result));
	}


	/* Allocates memory on the GPU for matrix operations pertaining to cases */

	ScalarBinary32 *cases_A_ptrGPU;
	result = cudaMalloc((ScalarBinary32 **) &cases_A_ptrGPU, sizeof(ScalarBinary32) * numSNPsWithPadding * (numCasesWithPadding / 32) * SNP_CALC);		
	if(result != cudaSuccess) {
		std::cerr << "Failed allocating memory for cases input data." << std::endl;
	}

	int *C_ptrGPU_cases;
	result = cudaMalloc((int**) &C_ptrGPU_cases, sizeof(int) * BATCH_SIZE * 2 * (BLOCK_SIZE * SNP_CALC) * (BLOCK_SIZE * SNP_CALC));        

	if(result != cudaSuccess) {
		std::cerr << "Failed allocating memory for cases output data." << std::endl;
	}


	/* Allocate memory on the GPU for matrix operations pertaining to controls */

	ScalarBinary32 *controls_A_ptrGPU;
	result = cudaMalloc((ScalarBinary32 **) &controls_A_ptrGPU, sizeof(ScalarBinary32) * numSNPsWithPadding * (numControlsWithPadding / 32) * SNP_CALC);	
	if(result != cudaSuccess) {
		std::cerr << "Failed allocating memory for controls input data." << std::endl;
	}

	int *C_ptrGPU_controls;
	result = cudaMalloc((int**) &C_ptrGPU_controls, sizeof(int) * BATCH_SIZE * 2 * (BLOCK_SIZE * SNP_CALC) * (BLOCK_SIZE * SNP_CALC));          

	if(result != cudaSuccess) {
		std::cerr << "Failed allocating memory for controls output data." << std::endl;
	}

	/* Computes a lgamma() lookup table on the Host that is to be used as part of K2 Bayasian scoring calculations on the GPU */
	float * d_tablePrecalc;
	int tablePrecalc_size = max(numCases, numControls) + 1;
	float * h_tablePrecalc = (float*) malloc(tablePrecalc_size * sizeof(float));
	for(int i=1; i < (tablePrecalc_size + 1); i++) {
		h_tablePrecalc[i - 1] = lgamma((double)i);
	}

	/* Alocates memory on the GPU for the lookup table and copies it from the Host to the GPU */
	result = cudaMalloc((float**)&d_tablePrecalc, tablePrecalc_size * sizeof(float));
	result = cudaMemcpy(d_tablePrecalc, h_tablePrecalc, tablePrecalc_size * sizeof(float), cudaMemcpyHostToDevice);


	float * d_output;
	unsigned long long int * d_output_packedIndices;

	/* Sets initial score to highest possible value, since K2 is a score to be minimized */
	float h_output[1] = {FLT_MAX};

	/* Allocates memory on GPU to store the best score and indexes of the corresponding set of SNPs, and sets the initial score. */
	result = cudaMalloc((float**)&d_output, 1 * sizeof(float));								
	result = cudaMalloc((unsigned long long int**)&d_output_packedIndices, 1 * sizeof(unsigned long long int));		
	result = cudaMemcpy(d_output, h_output, 1 * sizeof(float), cudaMemcpyHostToDevice);


	/* Allocates memory on the GPU for storing popcounts on the outcome of filtering the first super-block in relation to the first (X0) and second (X1) genotypes of a given SNP X, for cases and controls */

	uint * d_output_individualSNP_popcountsForCases_filteredBy_X0_super1;
	uint * d_output_individualSNP_popcountsForControls_filteredBy_X0_super1;

	uint * d_output_individualSNP_popcountsForCases_filteredBy_X1_super1;
	uint * d_output_individualSNP_popcountsForControls_filteredBy_X1_super1;

	result = cudaMalloc((uint**)&d_output_individualSNP_popcountsForControls_filteredBy_X0_super1, 3 * SUPERBLOCK_SIZE * sizeof(uint));
	result = cudaMalloc((uint**)&d_output_individualSNP_popcountsForCases_filteredBy_X0_super1, 3 * SUPERBLOCK_SIZE * sizeof(uint));

	result = cudaMalloc((uint**)&d_output_individualSNP_popcountsForControls_filteredBy_X1_super1, 3 * SUPERBLOCK_SIZE * sizeof(uint));
	result = cudaMalloc((uint**)&d_output_individualSNP_popcountsForCases_filteredBy_X1_super1, 3 * SUPERBLOCK_SIZE * sizeof(uint));

	/* Allocates memory on the GPU for storing popcounts on the the outcome of filtering the second super-block in relation to the first (X0) and second (X1) genotypes of a given SNP X, for cases and controls */

	uint * d_output_individualSNP_popcountsForCases_filteredBy_X0_super2;
	uint * d_output_individualSNP_popcountsForControls_filteredBy_X0_super2;

	uint * d_output_individualSNP_popcountsForCases_filteredBy_X1_super2;
	uint * d_output_individualSNP_popcountsForControls_filteredBy_X1_super2;

	result = cudaMalloc((uint**)&d_output_individualSNP_popcountsForControls_filteredBy_X0_super2, 3 * SUPERBLOCK_SIZE * sizeof(uint));
	result = cudaMalloc((uint**)&d_output_individualSNP_popcountsForCases_filteredBy_X0_super2, 3 * SUPERBLOCK_SIZE * sizeof(uint));

	result = cudaMalloc((uint**)&d_output_individualSNP_popcountsForControls_filteredBy_X1_super2, 3 * SUPERBLOCK_SIZE * sizeof(uint));
	result = cudaMalloc((uint**)&d_output_individualSNP_popcountsForCases_filteredBy_X1_super2, 3 * SUPERBLOCK_SIZE * sizeof(uint));


	/* Allocates memory on the GPU for the outcome of filtering the first super-block in relation to the first (X0) and second (X1) genotypes of a given SNP X, for cases and controls */

	uint * d_output_snpBlock_ForCases_X_super1;
	uint * d_output_snpBlock_ForControls_X_super1;
	result = cudaMalloc((uint**)&d_output_snpBlock_ForCases_X_super1, 2 * (numCasesWithPadding / 32) * (SNP_CALC) * SUPERBLOCK_SIZE * sizeof(uint));          
	if (result != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(result));
	} 
	result = cudaMalloc((uint**)&d_output_snpBlock_ForControls_X_super1, 2 * (numControlsWithPadding / 32) * (SNP_CALC) * SUPERBLOCK_SIZE * sizeof(uint));
	if (result != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(result));
	}

	/* Allocates memory on the GPU for the outcome of filtering the second super-block in relation to the first (X0) and second (X1) genotypes of a given SNP X, for cases and controls */

	uint * d_output_snpBlock_ForCases_X_super2;
	uint * d_output_snpBlock_ForControls_X_super2;
	result = cudaMalloc((uint**)&d_output_snpBlock_ForCases_X_super2, 2 * (numCasesWithPadding / 32) * (SNP_CALC) * SUPERBLOCK_SIZE * sizeof(uint));
	if (result != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(result));
	}
	result = cudaMalloc((uint**)&d_output_snpBlock_ForControls_X_super2, 2 * (numControlsWithPadding / 32) * (SNP_CALC) * SUPERBLOCK_SIZE * sizeof(uint));
	if (result != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(result));
	}


	/* For storing the popcounts related to SNP pairs resulting from combining two super-blocks. */
	uint * d_output_pairwiseSNP_popcountsForCases;
	uint * d_output_pairwiseSNP_popcountsForControls;
	result = cudaMalloc((uint**)&d_output_pairwiseSNP_popcountsForControls, 9LL * SUPERBLOCK_SIZE * SUPERBLOCK_SIZE * sizeof(uint));
	result = cudaMalloc((uint**)&d_output_pairwiseSNP_popcountsForCases, 9LL * SUPERBLOCK_SIZE * SUPERBLOCK_SIZE * sizeof(uint));


	/* CUDA stream creation */
	cudaStream_t cudaStreamPairwiseSNPs;
	cudaStreamCreate(&cudaStreamPairwiseSNPs);
	cudaStream_t cudaStreamforX;
	cudaStreamCreate(&cudaStreamforX);


	/* Lookup table that is used to store the amount of samples with the first (X0) and second (X1) genotypes in regard to a given SNP X, for cases and controls */
	uint *dimensionsPrunned_cases = (uint*) malloc(sizeof(uint) * numSNPs * 2);	
	uint *dimensionsPrunned_controls = (uint*) malloc(sizeof(uint) * numSNPs * 2);     

	/* Number of 32-bit bit-chunks of cases and controls (considering padding) */
	int casesSize = ceil(((float) numCases) / ((float) PADDING_SAMPLES)) * PADDING_SAMPLES / 32;
	int controlsSize = ceil(((float) numControls) / ((float) PADDING_SAMPLES)) * PADDING_SAMPLES / 32;


	/* Reorganizes the dataset internal representation to maximize the impact of data compaction based on genotype counts for individual SNPs. */

	unsigned int *datasetCases_temp = (unsigned int *) malloc(casesSize * sizeof(unsigned int));
        unsigned int *datasetControls_temp = (unsigned int *) malloc(controlsSize * sizeof(unsigned int));

	/* Used for creating the masks for dealing with the last bitpack */
        int casesSizeNoPadding = ceil(((float) numCases) / 32.0f);
        int controlsSizeNoPadding = ceil(((float) numControls) / 32.0f);

	#pragma omp parallel for
	for(int snp_x = 0; snp_x < numSNPs; snp_x++) {

		int gen0_cases = 0;
		int gen1_cases = 0;
		int gen2_cases = 0;
                
		for(int cases_i=0; cases_i < casesSize; cases_i++) {
			gen0_cases +=  __builtin_popcount(datasetCases_host_matrixA[snp_x * SNP_CALC * casesSize + 0 * casesSize + cases_i]);
                        gen1_cases +=  __builtin_popcount(datasetCases_host_matrixA[snp_x * SNP_CALC * casesSize + 1 * casesSize + cases_i]);
		}

		gen2_cases = numCases - gen0_cases - gen1_cases;

                /* Saves counts for using when compacting data */
                dimensionsPrunned_cases[0 * numSNPs + snp_x] = gen0_cases;
                dimensionsPrunned_cases[1 * numSNPs + snp_x] = gen1_cases;

		int gen0_controls = 0;
                int gen1_controls = 0;
                int gen2_controls = 0;

		for(int controls_i=0; controls_i < controlsSize; controls_i++) {
			gen0_controls +=  __builtin_popcount(datasetControls_host_matrixA[snp_x * SNP_CALC * controlsSize + 0 * controlsSize + controls_i]);
                        gen1_controls +=  __builtin_popcount(datasetControls_host_matrixA[snp_x * SNP_CALC * controlsSize + 1 * controlsSize + controls_i]);
		}

                gen2_controls = numControls - gen0_controls - gen1_controls;

		/* Saves counts for using when compacting data */
		dimensionsPrunned_controls[0 * numSNPs + snp_x] = gen0_controls;
		dimensionsPrunned_controls[1 * numSNPs + snp_x] = gen1_controls;

		int gen0_total = gen0_cases + gen0_controls;
                int gen1_total = gen1_cases + gen1_controls;
                int gen2_total = gen2_cases + gen2_controls;

		#if !defined(DISABLE_REORDERING)
		
		/* Reorders placement of genotype data in dataset representation to use during the search */

		/* Masks used to process the last bitpack taking into account the number of cases or controls might not be multiple of 32. */
		uint maskRelevantBitsSetCases = (~0u) << (casesSizeNoPadding * 32 - numCases);
		uint maskRelevantBitsSetControls = (~0u) << (controlsSizeNoPadding * 32 - numControls);

		/* Keeps track of genotype with lowest occurance */	
		int gen_first = 0;

                // printf("[counts before reordering] 0: %d, 1: %d, 2: %d\n", gen0_total, gen1_total, gen2_total);

		/* If genotype 1 has the lowest occurence rate, its data is transfered to the original position of genotype 0 data */
		if((gen1_total < gen0_total) && (gen1_total <= gen2_total)) {

			/* Copies genotype 0 data to temporary array */
			memcpy(datasetCases_temp, &(datasetCases_host_matrixA[snp_x * SNP_CALC * casesSize + 0 * casesSize]), casesSize * sizeof(unsigned int));
			memcpy(datasetControls_temp, &(datasetControls_host_matrixA[snp_x * SNP_CALC * controlsSize + 0 * controlsSize]), controlsSize * sizeof(unsigned int));

			/* Moves genotype 1 data to original position of genotype 0 data */
			memcpy(&(datasetCases_host_matrixA[snp_x * SNP_CALC * casesSize + 0 * casesSize]), &(datasetCases_host_matrixA[snp_x * SNP_CALC * casesSize + 1 * casesSize]), casesSize * sizeof(unsigned int));
			memcpy(&(datasetControls_host_matrixA[snp_x * SNP_CALC * controlsSize + 0 * controlsSize]), &(datasetControls_host_matrixA[snp_x * SNP_CALC * controlsSize + 1 * controlsSize]), controlsSize * sizeof(unsigned int));

			/* Updates genotype arrays with genotype counts */
			dimensionsPrunned_cases[0 * numSNPs + snp_x] = gen1_cases;
			dimensionsPrunned_controls[0 * numSNPs + snp_x] = gen1_controls;

			gen_first = 1;

		}

		/* If genotype 2 has the lowest occurence rate, its data is transfered to the original position of genotype 0 data */
		else if((gen2_total < gen1_total) && (gen2_total < gen0_total)) {

                        /* Copies genotype 0 data to temporary array */
                        memcpy(datasetCases_temp, &(datasetCases_host_matrixA[snp_x * SNP_CALC * casesSize + 0 * casesSize]), casesSize * sizeof(unsigned int));
                        memcpy(datasetControls_temp, &(datasetControls_host_matrixA[snp_x * SNP_CALC * controlsSize + 0 * controlsSize]), controlsSize * sizeof(unsigned int));

                        /* Reconstructs genotype 2 data and stores it in the original position of genotype 0 data */
                        for(int cases_i=0; cases_i < casesSize; cases_i++) {
                                datasetCases_host_matrixA[snp_x * SNP_CALC * casesSize + 0 * casesSize + cases_i] = ~(datasetCases_host_matrixA[snp_x * SNP_CALC * casesSize + 0 * casesSize + cases_i] | datasetCases_host_matrixA[snp_x * SNP_CALC * casesSize + 1 * casesSize + cases_i]);
                        }

                        /* Uses mask for processing the last bitpack */
                        datasetCases_host_matrixA[snp_x * SNP_CALC * casesSize + 0 * casesSize + (casesSize - 1)] &= maskRelevantBitsSetCases;

                        for(int controls_i=0; controls_i < controlsSize; controls_i++) {

                                datasetControls_host_matrixA[snp_x * SNP_CALC * controlsSize + 0 * controlsSize + controls_i] = ~(datasetControls_host_matrixA[snp_x * SNP_CALC * controlsSize + 0 * controlsSize + controls_i] | datasetControls_host_matrixA[snp_x * SNP_CALC * controlsSize + 1 * controlsSize + controls_i]);
                        }

                        /* Uses mask for processing the last bitpack */
			datasetControls_host_matrixA[snp_x * SNP_CALC * controlsSize + 0 * controlsSize + (controlsSize - 1)] &= maskRelevantBitsSetControls;

			/* Updates genotype arrays with genotype counts */
			dimensionsPrunned_cases[0 * numSNPs + snp_x] = gen2_cases;
			dimensionsPrunned_controls[0 * numSNPs + snp_x] = gen2_controls;

			gen_first = 2;
		}

		/* Deals with the second lowest count genotype based on which has been identified to be the rarest genotype */

		if(gen_first == 0) {

			if(gen2_total < gen1_total) {

				/* Reconstructs genotype 2 data and stores it in the original position of genotype 1 data */
				for(int cases_i=0; cases_i < casesSize; cases_i++) {
					datasetCases_host_matrixA[snp_x * SNP_CALC * casesSize + 1 * casesSize + cases_i] = ~(datasetCases_host_matrixA[snp_x * SNP_CALC * casesSize + 0 * casesSize + cases_i] | datasetCases_host_matrixA[snp_x * SNP_CALC * casesSize + 1 * casesSize + cases_i]);
				}

				/* Uses mask for processing the last bitpack */
				datasetCases_host_matrixA[snp_x * SNP_CALC * casesSize + 1 * casesSize + (casesSize - 1)] &= maskRelevantBitsSetCases;

				for(int controls_i=0; controls_i < controlsSize; controls_i++) {

					datasetControls_host_matrixA[snp_x * SNP_CALC * controlsSize + 1 * controlsSize + controls_i] = ~(datasetControls_host_matrixA[snp_x * SNP_CALC * controlsSize + 0 * controlsSize + controls_i] | datasetControls_host_matrixA[snp_x * SNP_CALC * controlsSize + 1 * controlsSize + controls_i]);
				}

				/* Uses mask for processing the last bitpack */
				datasetControls_host_matrixA[snp_x * SNP_CALC * controlsSize + 1 * controlsSize + (controlsSize - 1)] &= maskRelevantBitsSetControls;

				/* Updates genotype arrays with genotype counts */
				dimensionsPrunned_cases[1 * numSNPs + snp_x] = gen2_cases;
				dimensionsPrunned_controls[1 * numSNPs + snp_x] = gen2_controls;
			}
		}

                if(gen_first == 1) {
			
			if(gen2_total < gen0_total) {
				
                                /* Reconstructs genotype 2 data and stores it in the original position of genotype 1 data */
                                for(int cases_i=0; cases_i < casesSize; cases_i++) {
                                        datasetCases_host_matrixA[snp_x * SNP_CALC * casesSize + 1 * casesSize + cases_i] = ~(datasetCases_host_matrixA[snp_x * SNP_CALC * casesSize + 0 * casesSize + cases_i] | datasetCases_host_matrixA[snp_x * SNP_CALC * casesSize + 1 * casesSize + cases_i]);
                                }

                                /* Uses mask for processing the last bitpack */
                                datasetCases_host_matrixA[snp_x * SNP_CALC * casesSize + 1 * casesSize + (casesSize - 1)] &= maskRelevantBitsSetCases;

                                for(int controls_i=0; controls_i < controlsSize; controls_i++) {

                                        datasetControls_host_matrixA[snp_x * SNP_CALC * controlsSize + 1 * controlsSize + controls_i] = ~(datasetControls_host_matrixA[snp_x * SNP_CALC * controlsSize + 0 * controlsSize + controls_i] | datasetControls_host_matrixA[snp_x * SNP_CALC * controlsSize + 1 * controlsSize + controls_i]);
                                }

				/* Uses mask for processing the last bitpack */
				datasetControls_host_matrixA[snp_x * SNP_CALC * controlsSize + 1 * controlsSize + (controlsSize - 1)] &= maskRelevantBitsSetControls;

				/* Updates genotype arrays with genotype counts */
				dimensionsPrunned_cases[1 * numSNPs + snp_x] = gen2_cases;
				dimensionsPrunned_controls[1 * numSNPs + snp_x] = gen2_controls;
			}
			else {
				/* Moves genotype 0 data from temporary array to the original position of genotype 1 data */
				memcpy(&(datasetCases_host_matrixA[snp_x * SNP_CALC * casesSize + 1 * casesSize]), datasetCases_temp, casesSize * sizeof(unsigned int));
				memcpy(&(datasetControls_host_matrixA[snp_x * SNP_CALC * controlsSize + 1 * controlsSize]), datasetControls_temp, controlsSize * sizeof(unsigned int));

				/* Updates genotype arrays with genotype counts */
				dimensionsPrunned_cases[1 * numSNPs + snp_x] = gen0_cases;
				dimensionsPrunned_controls[1 * numSNPs + snp_x] = gen0_controls;

			}
		}

		if(gen_first == 2) {

			/* Checks if genotype 0 is to be stored in original position of genotype 1, doing nothing otherwise since genotype 1 data is already there. */
			if(gen0_total <= gen1_total) {
				
				/* Moves genotype 0 data from temporary array to the original position of genotype 1 data */
				memcpy(&(datasetCases_host_matrixA[snp_x * SNP_CALC * casesSize + 0 * casesSize]), datasetCases_temp, casesSize * sizeof(unsigned int));
				memcpy(&(datasetControls_host_matrixA[snp_x * SNP_CALC * controlsSize + 0 * controlsSize]), datasetControls_temp, controlsSize * sizeof(unsigned int));

				/* Updates genotype arrays with genotype counts */
				dimensionsPrunned_cases[1 * numSNPs + snp_x] = gen0_cases;
				dimensionsPrunned_controls[1 * numSNPs + snp_x] = gen0_controls;

			}

		}

		// printf("[counts after reordering] 0: %d, 1: %d, 2: %d\n", dimensionsPrunned_cases[0 * numSNPs + snp_x] + dimensionsPrunned_controls[0 * numSNPs + snp_x], dimensionsPrunned_cases[1 * numSNPs + snp_x] + dimensionsPrunned_controls[1 * numSNPs + snp_x], (numCases + numControls) - (dimensionsPrunned_cases[0 * numSNPs + snp_x] + dimensionsPrunned_controls[0 * numSNPs + snp_x]) - (dimensionsPrunned_cases[1 * numSNPs + snp_x] + dimensionsPrunned_controls[1 * numSNPs + snp_x]) );

		#endif
	}

	/* Copies cases and controls data from host to GPU. */
        result = cudaMemcpyAsync(cases_A_ptrGPU, datasetCases_host_matrixA, sizeof(int) * numSNPsWithPadding * (numCasesWithPadding / 32) * SNP_CALC, cudaMemcpyHostToDevice, 0);
        result = cudaMemcpyAsync(controls_A_ptrGPU, datasetControls_host_matrixA, sizeof(int) * numSNPsWithPadding * (numControlsWithPadding / 32) * SNP_CALC, cudaMemcpyHostToDevice, 0);

	MPI_Status stat;
	int start_Y;
	int X_index;
	int start_Z;

	unsigned long long int start_Y_and_X_index_and_start_Z;


	/* Counter for keeping track of the iteration being processed (maximum of NUM_ITER) */
	int iter;

	/* Keeps track if the first and the second super-blocks have been processed in the previous iteration.
	   This is used to avoid having to recompute the second-order genotype counts resulting from combining the superblocks.  */
	int prev_superblock_Y_start = -1;
	int prev_superblock_Z_start = -1;


	/* Loop that iterates until the MPI worker is not assigned more work by the MPI coordinator */
	while (1) {


		struct timespec t_iter_start, t_iter_end;
		clock_gettime(CLOCK_MONOTONIC, &t_iter_start);       // initial timestamp

		MPI_Send (NULL, 0 , MPI_INT, 0 /* goes to rank 0 */, 0, MPI_COMM_WORLD);	

		/* Tries to get new unit of work from coordinator */
		MPI_Recv (&start_Y_and_X_index_and_start_Z , 1, MPI_UNSIGNED_LONG_LONG, 0 /* comes from rank 0 */, 0, MPI_COMM_WORLD, &stat);

		/* Breaks the while loop if there is no more data to process */
		if (start_Y_and_X_index_and_start_Z == 0xffffffffffffffff) {	
			// printf("Worker with MPI rank %d was informed there is no more data to process.\n", mpiRank);
			break;
		}

		/* Unpacks indices representing first evaluation round to perform as part of the unit of work */
		start_Y = start_Y_and_X_index_and_start_Z & 0xFFFFF;
		X_index = (start_Y_and_X_index_and_start_Z >> 20) & 0xFFFFF;
		start_Z = (start_Y_and_X_index_and_start_Z >> 40) & 0xFFFFF;

		/* Resets the iteration counter */
		iter = 0;

		/* Calculates the super-block indices from the block indices */
		unsigned int superblock_Y_start = floor((double)start_Y / (double)SUPERBLOCK_SIZE) * SUPERBLOCK_SIZE;	
		unsigned int superblock_Z_start = floor((double)start_Z / (double)SUPERBLOCK_SIZE) * SUPERBLOCK_SIZE;


		/* Iterates SNP blocks inside the first super-block (Y) until all have been considered or the target number of evaluation rounds has been achieved */
		while((superblock_Y_start < numSNPsWithPadding) && (iter < NUM_ITER)) {

			/* Iterates SNP blocks inside the second super-block (Z) until all have been considered or the target number of evaluation rounds has been achieved */
			while((superblock_Z_start < numSNPsWithPadding) && (iter < NUM_ITER)) {

				// printf("super-block {0 start, 1 start}: { %d , %d }\n", superblock_Y_start, superblock_Z_start);

				dim3 blocksPerGrid_pairwise ( (size_t)ceil(((float)(SUPERBLOCK_SIZE)) / ((float)8)), (size_t)ceil(((float)(SUPERBLOCK_SIZE)) / ((float)32)), 1 );
				dim3 workgroupSize_pairwise ( 8, 32, 1 );	

				/* Constructs second-order contingency tables resulting from combining the super-blocks, but only if it has not already been done before */
				if (((prev_superblock_Y_start != superblock_Y_start) || (prev_superblock_Z_start != superblock_Z_start))) {	

					pairPop_superblocks<<<blocksPerGrid_pairwise, workgroupSize_pairwise, 0, cudaStreamPairwiseSNPs>>>((uint*)cases_A_ptrGPU, (uint*)controls_A_ptrGPU, d_output_pairwiseSNP_popcountsForCases, d_output_pairwiseSNP_popcountsForControls, numSNPs, numCases, numControls, superblock_Y_start, superblock_Z_start);

					/* Saves information on which superblocks have been used to construct the second-order tables */
					prev_superblock_Y_start = superblock_Y_start;
					prev_superblock_Z_start = superblock_Z_start;
				}	


				while((X_index < (superblock_Y_start + SUPERBLOCK_SIZE)) && (iter < NUM_ITER)) {

					/* Gets the number of padded samples after filtering */
					uint numCasesAfterPrunning_X[2];        
					uint numControlsAfterPrunning_X[2];     
					numCasesAfterPrunning_X[0] = (int) ceil((double) dimensionsPrunned_cases[0 * numSNPs + X_index] /  PADDING_SAMPLES) * PADDING_SAMPLES;
					numCasesAfterPrunning_X[1] = (int) ceil((double) dimensionsPrunned_cases[1 * numSNPs + X_index] / PADDING_SAMPLES) * PADDING_SAMPLES;
					numControlsAfterPrunning_X[0] = (int) ceil((double) dimensionsPrunned_controls[0 * numSNPs + X_index] / PADDING_SAMPLES) * PADDING_SAMPLES;
					numControlsAfterPrunning_X[1] = (int) ceil((double) dimensionsPrunned_controls[1 * numSNPs + X_index] / PADDING_SAMPLES) * PADDING_SAMPLES;

					/* Same as above, but without padding.
					   The individual popcount kernel needs information on the exact number of cases and controls (i.e. without padding) */
					uint numCasesAfterPrunning_X_noPadding[2];
					uint numControlsAfterPrunning_X_noPadding[2];
					numCasesAfterPrunning_X_noPadding[0] = dimensionsPrunned_cases[0 * numSNPs + X_index];
					numCasesAfterPrunning_X_noPadding[1] = dimensionsPrunned_cases[1 * numSNPs + X_index];
					numControlsAfterPrunning_X_noPadding[0] = dimensionsPrunned_controls[0 * numSNPs + X_index];
					numControlsAfterPrunning_X_noPadding[1] = dimensionsPrunned_controls[1 * numSNPs + X_index];

					int numColsInts_cases =  (numCases + 31) / 32;
					int numColsInts_controls =   (numControls + 31) / 32;

					int threadsPerBlock = 256;	
					int blocksPerGrid_cases = (numColsInts_cases + threadsPerBlock - 1) / threadsPerBlock;
					int blocksPerGrid_controls = (numColsInts_controls + threadsPerBlock - 1) / threadsPerBlock;


					/* Resets popcount arrays pertaining to X0 and X1 for cases and controls */ 
					cudaMemsetAsync(d_counts_X0_cases, 0, numColsInts_cases * sizeof(unsigned int), cudaStreamforX);
					cudaMemsetAsync(d_counts_X1_cases, 0, numColsInts_cases * sizeof(unsigned int), cudaStreamforX);
					cudaMemsetAsync(d_counts_X0_controls, 0, numColsInts_controls * sizeof(unsigned int), cudaStreamforX);
					cudaMemsetAsync(d_counts_X1_controls, 0, numColsInts_controls * sizeof(unsigned int), cudaStreamforX);

					/* Launches instances of kernel that counts set bits, at the granularity of 32-bit bit-packs */
					countBits<<<blocksPerGrid_cases, threadsPerBlock, 0, cudaStreamforX>>>(&(((uint*)cases_A_ptrGPU)[((X_index * 2) + 0) * numColsInts_cases]), d_counts_X0_cases, numColsInts_cases);
					countBits<<<blocksPerGrid_cases, threadsPerBlock, 0, cudaStreamforX>>>(&(((uint*)cases_A_ptrGPU)[((X_index * 2) + 1) * numColsInts_cases]), d_counts_X1_cases, numColsInts_cases);
					countBits<<<blocksPerGrid_controls, threadsPerBlock, 0, cudaStreamforX>>>(&(((uint*)controls_A_ptrGPU)[((X_index * 2) + 0) * numColsInts_controls]), d_counts_X0_controls, numColsInts_controls);
					countBits<<<blocksPerGrid_controls, threadsPerBlock, 0, cudaStreamforX>>>(&(((uint*)controls_A_ptrGPU)[((X_index * 2) + 1) * numColsInts_controls]), d_counts_X1_controls, numColsInts_controls);

					cudaError_t err = cudaGetLastError();
					if (err != cudaSuccess) {
						fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
						exit(EXIT_FAILURE);
					}

					unsigned int * ptr_d_output_snpBlock_ForCases_X_super1;
					unsigned int * ptr_d_output_snpBlock_ForControls_X_super1;
					unsigned int * ptr_d_output_snpBlock_ForCases_X_super2;
					unsigned int * ptr_d_output_snpBlock_ForControls_X_super2;

					ptr_d_output_snpBlock_ForCases_X_super1 = d_output_snpBlock_ForCases_X_super1;
					ptr_d_output_snpBlock_ForControls_X_super1 = d_output_snpBlock_ForControls_X_super1;

					dim3 blocksPerGrid_compact_bits_cases_super1(((SUPERBLOCK_SIZE - (start_Y - superblock_Y_start)) * 2) / SNPS_PER_THREAD, (numColsInts_cases + threadsPerBlock - 1) / threadsPerBlock, 1);
					dim3 blocksPerGrid_compact_bits_controls_super1(((SUPERBLOCK_SIZE - (start_Y - superblock_Y_start)) * 2) / SNPS_PER_THREAD, (numColsInts_controls + threadsPerBlock - 1) / threadsPerBlock, 1); 


					dim3 blocksPerGrid_compact_bits_cases_super2(((SUPERBLOCK_SIZE - (start_Z - superblock_Z_start)) * 2) / SNPS_PER_THREAD, (numColsInts_cases + threadsPerBlock - 1) / threadsPerBlock, 1);
					dim3 blocksPerGrid_compact_bits_controls_super2(((SUPERBLOCK_SIZE - (start_Z - superblock_Z_start)) * 2) / SNPS_PER_THREAD, (numColsInts_controls + threadsPerBlock - 1) / threadsPerBlock, 1);


					dim3 workgroupGrid_compact_bits(1, threadsPerBlock, 1);


					/* Performs the prefix sum for the cases using the CUB library */

					if(d_tempStorage_x0_cases == NULL) {
						cub::DeviceScan::InclusiveSum(d_tempStorage_x0_cases, tempStorageSize_x0_cases, d_counts_X0_cases, d_prefixSum_X0_cases, numColsInts_cases, cudaStreamforX);
						cudaMalloc(&d_tempStorage_x0_cases, tempStorageSize_x0_cases);
					}
					cub::DeviceScan::InclusiveSum(d_tempStorage_x0_cases, tempStorageSize_x0_cases, d_counts_X0_cases, d_prefixSum_X0_cases, numColsInts_cases, cudaStreamforX);

					err = cudaGetLastError();	
					if (err != cudaSuccess) {
						fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
						exit(EXIT_FAILURE);
					}

					if(d_tempStorage_x1_cases == NULL) {
						cub::DeviceScan::InclusiveSum(d_tempStorage_x1_cases, tempStorageSize_x1_cases, d_counts_X1_cases, d_prefixSum_X1_cases, numColsInts_cases, cudaStreamforX);
						cudaMalloc(&d_tempStorage_x1_cases, tempStorageSize_x1_cases);
					}
					cub::DeviceScan::InclusiveSum(d_tempStorage_x1_cases, tempStorageSize_x1_cases, d_counts_X1_cases, d_prefixSum_X1_cases, numColsInts_cases, cudaStreamforX);

					err = cudaGetLastError();
					if (err != cudaSuccess) {
						fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
						exit(EXIT_FAILURE);
					}


					/* Performs the prefix sum for the controls using the CUB library */

					if(d_tempStorage_x0_controls == NULL) {
						cub::DeviceScan::InclusiveSum(d_tempStorage_x0_controls, tempStorageSize_x0_controls, d_counts_X0_controls, d_prefixSum_X0_controls, numColsInts_controls, cudaStreamforX);
						cudaMalloc(&d_tempStorage_x0_controls, tempStorageSize_x0_controls);
					}
					cub::DeviceScan::InclusiveSum(d_tempStorage_x0_controls, tempStorageSize_x0_controls, d_counts_X0_controls, d_prefixSum_X0_controls, numColsInts_controls, cudaStreamforX);
					err = cudaGetLastError();

					if (err != cudaSuccess) {
						fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
						exit(EXIT_FAILURE);
					}

					if(d_tempStorage_x1_controls == NULL) {
						cub::DeviceScan::InclusiveSum(d_tempStorage_x1_controls, tempStorageSize_x1_controls, d_counts_X1_controls, d_prefixSum_X1_controls, numColsInts_controls, cudaStreamforX);
						cudaMalloc(&d_tempStorage_x1_controls, tempStorageSize_x1_controls);
					}
					cub::DeviceScan::InclusiveSum(d_tempStorage_x1_controls, tempStorageSize_x1_controls, d_counts_X1_controls, d_prefixSum_X1_controls, numColsInts_controls, cudaStreamforX);

					err = cudaGetLastError();
					if (err != cudaSuccess) {
						fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
						exit(EXIT_FAILURE);
					}



					/* Clears memory for saving the first super-block (Y) after filtering in relation to X0 and X1, for cases and controls */
					cudaMemsetAsync(ptr_d_output_snpBlock_ForCases_X_super1, 0, SUPERBLOCK_SIZE * SNP_CALC * numColsInts_cases * sizeof(unsigned int), cudaStreamforX);
					cudaMemsetAsync(ptr_d_output_snpBlock_ForCases_X_super1 + (SUPERBLOCK_SIZE * SNP_CALC * numColsInts_cases), 0, SUPERBLOCK_SIZE * SNP_CALC * numColsInts_cases * sizeof(unsigned int), cudaStreamforX);
					cudaMemsetAsync(ptr_d_output_snpBlock_ForControls_X_super1, 0, SUPERBLOCK_SIZE * SNP_CALC * numColsInts_controls * sizeof(unsigned int), cudaStreamforX);
					cudaMemsetAsync(ptr_d_output_snpBlock_ForControls_X_super1 + (SUPERBLOCK_SIZE * SNP_CALC * numColsInts_controls), 0, SUPERBLOCK_SIZE * SNP_CALC * numColsInts_controls * sizeof(unsigned int), cudaStreamforX);


					/* Filters out cases that do not have the first genotype (0) of the SNP X being processed */
					compactBits<<<blocksPerGrid_compact_bits_cases_super1, workgroupGrid_compact_bits, 0, cudaStreamforX>>>((uint*)cases_A_ptrGPU, ptr_d_output_snpBlock_ForCases_X_super1, d_prefixSum_X0_cases, numColsInts_cases, (X_index * 2) + 0, (numCasesAfterPrunning_X[0] + 31) / 32, (start_Y - superblock_Y_start));



					/* Filters out cases that do not have the second genotype (1) of the SNP X being processed */
					compactBits<<<blocksPerGrid_compact_bits_cases_super1, workgroupGrid_compact_bits, 0, cudaStreamforX>>>((uint*)cases_A_ptrGPU, ptr_d_output_snpBlock_ForCases_X_super1 + (SUPERBLOCK_SIZE * SNP_CALC * numColsInts_cases), d_prefixSum_X1_cases, numColsInts_cases, (X_index * 2) + 1, (numCasesAfterPrunning_X[1] + 31) / 32, (start_Y - superblock_Y_start));

                                        /* Filters out controls that do not have the first genotype (0) of the SNP X being processed */
					compactBits<<<blocksPerGrid_compact_bits_controls_super1, workgroupGrid_compact_bits, 0, cudaStreamforX>>>((uint*)controls_A_ptrGPU, ptr_d_output_snpBlock_ForControls_X_super1, d_prefixSum_X0_controls, numColsInts_controls, (X_index * 2) + 0, (numControlsAfterPrunning_X[0] + 31) / 32, (start_Y - superblock_Y_start));


                                        /* Filters out controls that do not have the second genotype (1) of the SNP X being processed */
					compactBits<<<blocksPerGrid_compact_bits_controls_super1, workgroupGrid_compact_bits, 0, cudaStreamforX>>>((uint*)controls_A_ptrGPU, ptr_d_output_snpBlock_ForControls_X_super1 + (SUPERBLOCK_SIZE * SNP_CALC * numColsInts_controls), d_prefixSum_X1_controls, numColsInts_controls, (X_index * 2) + 1, (numControlsAfterPrunning_X[1] + 31) / 32, (start_Y - superblock_Y_start));


					/* Filters the second super-block in relation to X0 and X1, but only if it is different from the first super-block */
					if(superblock_Y_start != superblock_Z_start) {

						ptr_d_output_snpBlock_ForCases_X_super2 = d_output_snpBlock_ForCases_X_super2;
						ptr_d_output_snpBlock_ForControls_X_super2 = d_output_snpBlock_ForControls_X_super2;

                                        	/* Clears memory for saving the second super-block (Z) after filtering in relation to X0 and X1, for cases and controls */
						cudaMemsetAsync(ptr_d_output_snpBlock_ForCases_X_super2, 0, SUPERBLOCK_SIZE * SNP_CALC * numColsInts_cases * sizeof(unsigned int), cudaStreamforX);
						cudaMemsetAsync(ptr_d_output_snpBlock_ForCases_X_super2 + (SUPERBLOCK_SIZE * SNP_CALC * numColsInts_cases), 0, SUPERBLOCK_SIZE * SNP_CALC * numColsInts_cases * sizeof(unsigned int), cudaStreamforX);
						cudaMemsetAsync(ptr_d_output_snpBlock_ForControls_X_super2, 0, SUPERBLOCK_SIZE * SNP_CALC * numColsInts_controls * sizeof(unsigned int), cudaStreamforX);
						cudaMemsetAsync(ptr_d_output_snpBlock_ForControls_X_super2 + (SUPERBLOCK_SIZE * SNP_CALC * numColsInts_controls), 0, SUPERBLOCK_SIZE * SNP_CALC * numColsInts_controls * sizeof(unsigned int), cudaStreamforX);

						/* Filters out cases that do not have the first genotype (0) of the SNP X being processed */
						compactBits<<<blocksPerGrid_compact_bits_cases_super2, workgroupGrid_compact_bits, 0, cudaStreamforX>>>((uint*)cases_A_ptrGPU, ptr_d_output_snpBlock_ForCases_X_super2, d_prefixSum_X0_cases, numColsInts_cases, (X_index * 2) + 0, (numCasesAfterPrunning_X[0] + 31) / 32, (start_Z - superblock_Z_start));

						/* Filters out cases that do not have the second genotype (1) of the SNP X being processed */
						compactBits<<<blocksPerGrid_compact_bits_cases_super2, workgroupGrid_compact_bits, 0, cudaStreamforX>>>((uint*)cases_A_ptrGPU, ptr_d_output_snpBlock_ForCases_X_super2 + (SUPERBLOCK_SIZE * SNP_CALC * numColsInts_cases), d_prefixSum_X1_cases, numColsInts_cases, (X_index * 2) + 1, (numCasesAfterPrunning_X[1] + 31) / 32, (start_Z - superblock_Z_start));


						/* Filters out controls that do not have the first genotype (0) of the SNP X being processed */
						compactBits<<<blocksPerGrid_compact_bits_controls_super2, workgroupGrid_compact_bits, 0, cudaStreamforX>>>((uint*)controls_A_ptrGPU, ptr_d_output_snpBlock_ForControls_X_super2, d_prefixSum_X0_controls, numColsInts_controls, (X_index * 2) + 0, (numControlsAfterPrunning_X[0] + 31) / 32, (start_Z - superblock_Z_start));

						/* Filters out controls that do not have the second genotype (1) of the SNP X being processed */
						compactBits<<<blocksPerGrid_compact_bits_controls_super2, workgroupGrid_compact_bits, 0, cudaStreamforX>>>((uint*)controls_A_ptrGPU, ptr_d_output_snpBlock_ForControls_X_super2 + (SUPERBLOCK_SIZE * SNP_CALC * numColsInts_controls), d_prefixSum_X1_controls, numColsInts_controls, (X_index * 2) + 1, (numControlsAfterPrunning_X[1] + 31) / 32, (start_Z - superblock_Z_start));

					}
					/* In case both are the same super-block, then just copy pointers. */
					else {  

						ptr_d_output_snpBlock_ForCases_X_super2 = ptr_d_output_snpBlock_ForCases_X_super1;
						ptr_d_output_snpBlock_ForControls_X_super2 = ptr_d_output_snpBlock_ForControls_X_super1;
					}

					dim3 blocksPerGrid_ind_super1(SUPERBLOCK_SIZE, 1, 1);  
					dim3 blocksPerGrid_ind_super2(SUPERBLOCK_SIZE, 1, 1);
					dim3 workgroupSize_ind( 1, 256, 1 ); 


					/* Calculates second-order genotype counts for cases and controls that include X0 through counting bits set in {SNP,genotype} bitvectors after filtering the first super-block (Y) */
					pairPop<<<blocksPerGrid_ind_super1, workgroupSize_ind, 0, cudaStreamforX>>>(superblock_Y_start, ptr_d_output_snpBlock_ForCases_X_super1, ptr_d_output_snpBlock_ForControls_X_super1, d_output_individualSNP_popcountsForCases_filteredBy_X0_super1, d_output_individualSNP_popcountsForControls_filteredBy_X0_super1, numSNPs, numCasesAfterPrunning_X_noPadding[0], numControlsAfterPrunning_X_noPadding[0]);
					/* Calculates second-order genotype counts for cases and controls that include X1 through counting bits set in {SNP,genotype} bitvectors after filtering the first super-block (Y)*/
					pairPop<<<blocksPerGrid_ind_super1, workgroupSize_ind, 0, cudaStreamforX>>>(superblock_Y_start, ptr_d_output_snpBlock_ForCases_X_super1 + (SUPERBLOCK_SIZE * SNP_CALC * numColsInts_cases), ptr_d_output_snpBlock_ForControls_X_super1 + (SUPERBLOCK_SIZE * SNP_CALC * numColsInts_controls), d_output_individualSNP_popcountsForCases_filteredBy_X1_super1, d_output_individualSNP_popcountsForControls_filteredBy_X1_super1, numSNPs, numCasesAfterPrunning_X_noPadding[1], numControlsAfterPrunning_X_noPadding[1]);

					/* Calculates second-order genotype counts for cases and controls that include X0 through counting bits set in {SNP,genotype} bitvectors after filtering the second super-block (Z) */
					pairPop<<<blocksPerGrid_ind_super2, workgroupSize_ind, 0, cudaStreamforX>>>(superblock_Z_start, ptr_d_output_snpBlock_ForCases_X_super2, ptr_d_output_snpBlock_ForControls_X_super2, d_output_individualSNP_popcountsForCases_filteredBy_X0_super2, d_output_individualSNP_popcountsForControls_filteredBy_X0_super2, numSNPs, numCasesAfterPrunning_X_noPadding[0], numControlsAfterPrunning_X_noPadding[0]);
					/* Calculates second-order genotype counts for cases and controls that include X1 through counting bits set in {SNP,genotype} bitvectors after filtering the second super-block (Z) */
					pairPop<<<blocksPerGrid_ind_super2, workgroupSize_ind, 0, cudaStreamforX>>>(superblock_Z_start, ptr_d_output_snpBlock_ForCases_X_super2 + (SUPERBLOCK_SIZE * SNP_CALC * numColsInts_cases), ptr_d_output_snpBlock_ForControls_X_super2 + (SUPERBLOCK_SIZE * SNP_CALC * numColsInts_controls), d_output_individualSNP_popcountsForCases_filteredBy_X1_super2, d_output_individualSNP_popcountsForControls_filteredBy_X1_super2, numSNPs, numCasesAfterPrunning_X_noPadding[1], numControlsAfterPrunning_X_noPadding[1]);



					/* Counter that keeps track of batches during preparations for calling the batched AND+POPC matrix operation */
					int numBatches = 0;

					/* Stores pairings between SNP blocks from the first super-block (Y) with SNP-blocks from the second super-block (Z) */
					std::vector<int> start_Y_indexes_vec;
					std::vector<int> start_Z_indexes_vec;


					/*  */
					while((start_Y < (superblock_Y_start + SUPERBLOCK_SIZE)) && (iter < NUM_ITER)) {

						while((start_Z < (superblock_Z_start + SUPERBLOCK_SIZE)) && (iter < NUM_ITER)) {

							start_Y_indexes_vec.push_back(start_Y);
							start_Z_indexes_vec.push_back(start_Z);

							/* Processes first (X0) and second (X1) genotypes */
							for(int snp_x_genotype = 0; snp_x_genotype < 2; snp_x_genotype++) {

								uint numCasesAfterPrunning_X_forGenotype = numCasesAfterPrunning_X[snp_x_genotype];
								uint numControlsAfterPrunning_X_forGenotype = numControlsAfterPrunning_X[snp_x_genotype];

								unsigned int Y_fromStartSuperchunk = start_Y - superblock_Y_start;
								unsigned int Z_fromStartSuperchunk = start_Z - superblock_Z_start;

                                                                /* Processes cases */

                                                                ScalarBinary32 *A_ptrGPU_iter_cases = (ScalarBinary32 *) (ptr_d_output_snpBlock_ForCases_X_super1 + snp_x_genotype * (SUPERBLOCK_SIZE * SNP_CALC * (numCasesWithPadding / 32)) + (Y_fromStartSuperchunk * SNP_CALC * (numCasesAfterPrunning_X_forGenotype / 32)));

                                                                ScalarBinary32 *B_ptrGPU_iter_cases = (ScalarBinary32 *) (ptr_d_output_snpBlock_ForCases_X_super2 + snp_x_genotype * (SUPERBLOCK_SIZE * SNP_CALC * (numCasesWithPadding / 32)) + (Z_fromStartSuperchunk * SNP_CALC * (numCasesAfterPrunning_X_forGenotype / 32)));


                                                                host_A_array_cases[snp_x_genotype * MAX_BATCH + numBatches] = A_ptrGPU_iter_cases;
                                                                host_B_array_cases[snp_x_genotype * MAX_BATCH + numBatches] = B_ptrGPU_iter_cases;
                                                                host_C_array_cases[snp_x_genotype * MAX_BATCH + numBatches] = C_ptrGPU_cases + (numBatches % BATCH_SIZE) * 2 * ((BLOCK_SIZE * SNP_CALC) * (BLOCK_SIZE * SNP_CALC)) + snp_x_genotype * ((BLOCK_SIZE * SNP_CALC) * (BLOCK_SIZE * SNP_CALC));



                                                                /* Processes controls */

                                                                ScalarBinary32 *A_ptrGPU_iter_controls = (ScalarBinary32 *) (ptr_d_output_snpBlock_ForControls_X_super1 + snp_x_genotype * (SUPERBLOCK_SIZE * SNP_CALC * (numControlsWithPadding / 32)) + (Y_fromStartSuperchunk * SNP_CALC * (numControlsAfterPrunning_X_forGenotype / 32)));
                                                                ScalarBinary32 *B_ptrGPU_iter_controls = (ScalarBinary32 *) (ptr_d_output_snpBlock_ForControls_X_super2 + snp_x_genotype * (SUPERBLOCK_SIZE * SNP_CALC * (numControlsWithPadding / 32)) + (Z_fromStartSuperchunk * SNP_CALC * (numControlsAfterPrunning_X_forGenotype / 32)));

                                                                host_A_array_controls[snp_x_genotype * MAX_BATCH + numBatches] = A_ptrGPU_iter_controls;
                                                                host_B_array_controls[snp_x_genotype * MAX_BATCH + numBatches] = B_ptrGPU_iter_controls;
                                                                host_C_array_controls[snp_x_genotype * MAX_BATCH + numBatches] = C_ptrGPU_controls + (numBatches % BATCH_SIZE) * 2 * ((BLOCK_SIZE * SNP_CALC) * (BLOCK_SIZE * SNP_CALC)) + snp_x_genotype * ((BLOCK_SIZE * SNP_CALC) * (BLOCK_SIZE * SNP_CALC));

							}

							numBatches += 1;
							start_Z+=BLOCK_SIZE;
							iter += 1;
						}

						start_Y += BLOCK_SIZE;
						start_Z = MAX(start_Y, superblock_Z_start);	
					}

                                        /* Copies pointers for matrices A, B and C that pertain to the first genotype (X0) and to cases */
                                        result = cudaMemcpyAsync(ptr_A_array_cases + 0 * MAX_BATCH, host_A_array_cases + 0 * MAX_BATCH, numBatches * sizeof(void const *), cudaMemcpyHostToDevice, cudaStreamforX);
                                        result = cudaMemcpyAsync(ptr_B_array_cases + 0 * MAX_BATCH, host_B_array_cases + 0 * MAX_BATCH, numBatches * sizeof(void const *), cudaMemcpyHostToDevice, cudaStreamforX);
                                        result = cudaMemcpyAsync(ptr_C_array_cases + 0 * MAX_BATCH, host_C_array_cases + 0 * MAX_BATCH, numBatches * sizeof(void *), cudaMemcpyHostToDevice, cudaStreamforX);

                                        /* Copies pointers for matrices A, B and C that pertain to the first genotype (X1) and to cases */
                                        result = cudaMemcpyAsync(ptr_A_array_cases + 1 * MAX_BATCH, host_A_array_cases + 1 * MAX_BATCH, numBatches * sizeof(void const *), cudaMemcpyHostToDevice, cudaStreamforX);
                                        result = cudaMemcpyAsync(ptr_B_array_cases + 1 * MAX_BATCH, host_B_array_cases + 1 * MAX_BATCH, numBatches * sizeof(void const *), cudaMemcpyHostToDevice, cudaStreamforX);
                                        result = cudaMemcpyAsync(ptr_C_array_cases + 1 * MAX_BATCH, host_C_array_cases + 1 * MAX_BATCH, numBatches * sizeof(void *), cudaMemcpyHostToDevice, cudaStreamforX);

                                        /* Copies pointers for matrices A, B and C that pertain to the first genotype (X0) and to controls */
                                        result = cudaMemcpyAsync(ptr_A_array_controls + 0 * MAX_BATCH, host_A_array_controls + 0 * MAX_BATCH, numBatches * sizeof(void const *), cudaMemcpyHostToDevice, cudaStreamforX);
                                        result = cudaMemcpyAsync(ptr_B_array_controls + 0 * MAX_BATCH, host_B_array_controls + 0 * MAX_BATCH, numBatches * sizeof(void const *), cudaMemcpyHostToDevice, cudaStreamforX);
                                        result = cudaMemcpyAsync(ptr_C_array_controls + 0 * MAX_BATCH, host_C_array_controls + 0 * MAX_BATCH, numBatches * sizeof(void *), cudaMemcpyHostToDevice, cudaStreamforX);

                                        /* Copies pointers for matrices A, B and C that pertain to the first genotype (X1) and to controls */
                                        result = cudaMemcpyAsync(ptr_A_array_controls + 1 * MAX_BATCH, host_A_array_controls + 1 * MAX_BATCH, numBatches * sizeof(void const *), cudaMemcpyHostToDevice, cudaStreamforX);
                                        result = cudaMemcpyAsync(ptr_B_array_controls + 1 * MAX_BATCH, host_B_array_controls + 1 * MAX_BATCH, numBatches * sizeof(void const *), cudaMemcpyHostToDevice, cudaStreamforX);
                                        result = cudaMemcpyAsync(ptr_C_array_controls + 1 * MAX_BATCH, host_C_array_controls + 1 * MAX_BATCH, numBatches * sizeof(void *), cudaMemcpyHostToDevice, cudaStreamforX);


					for(int currentBatchIdx = 0; currentBatchIdx < numBatches; currentBatchIdx += BATCH_SIZE) {

						int batchesToProcess = BATCH_SIZE;

						if((numBatches - currentBatchIdx) < BATCH_SIZE) {
							batchesToProcess = numBatches - currentBatchIdx;
						}

						for(int snp_x_genotype = 0; snp_x_genotype < 2; snp_x_genotype++) {

							uint numCasesAfterPrunning_X_forGenotype = numCasesAfterPrunning_X[snp_x_genotype];     
							uint numControlsAfterPrunning_X_forGenotype = numControlsAfterPrunning_X[snp_x_genotype];       

                                                        result = tensorPopAnd_batched(
                                                                        BLOCK_SIZE * SNP_CALC,
                                                                        BLOCK_SIZE * SNP_CALC,
                                                                        numCasesAfterPrunning_X_forGenotype,
                                                                        numCasesAfterPrunning_X_forGenotype,
                                                                        numCasesAfterPrunning_X_forGenotype,
                                                                        BLOCK_SIZE * SNP_CALC,
                                                                        cudaStreamforX,
                                                                        ptr_A_array_cases + snp_x_genotype * MAX_BATCH + currentBatchIdx,
                                                                        ptr_B_array_cases + snp_x_genotype * MAX_BATCH + currentBatchIdx,
                                                                        ptr_C_array_cases + snp_x_genotype * MAX_BATCH + currentBatchIdx,
                                                                        batchesToProcess
                                                                        );

                                                        if(result != cudaSuccess) {
                                                                printf("Problem executing matmul\n");
                                                                return result;
                                                        }

                                                        result = tensorPopAnd_batched(
                                                                        BLOCK_SIZE * SNP_CALC,
                                                                        BLOCK_SIZE * SNP_CALC,
                                                                        numControlsAfterPrunning_X_forGenotype,
                                                                        numControlsAfterPrunning_X_forGenotype,
                                                                        numControlsAfterPrunning_X_forGenotype,
                                                                        BLOCK_SIZE * SNP_CALC,
                                                                        cudaStreamforX,
                                                                        ptr_A_array_controls + snp_x_genotype * MAX_BATCH + currentBatchIdx,
                                                                        ptr_B_array_controls + snp_x_genotype * MAX_BATCH + currentBatchIdx,
                                                                        ptr_C_array_controls + snp_x_genotype * MAX_BATCH + currentBatchIdx,
                                                                        batchesToProcess
                                                                        );

                                                        if(result != cudaSuccess) {
                                                                printf("Problem executing matmul\n");
                                                                return result;
                                                        }

						}

						/* Waits for pairwise genotype counts resulting from combining the super-blocks Y and Z to be processed */
						if(currentBatchIdx == 0) {
							cudaStreamSynchronize(cudaStreamPairwiseSNPs);
						}

						dim3 blocksPerGrid_objFun( (size_t)ceil(((float)(BLOCK_SIZE)) / ((float)BLOCK_OBJFUN) / ((float)1) ), (size_t)ceil(((float)(BLOCK_SIZE)) / ((float)256)), 1);
						dim3 workgroupSize_objFun( 1, 256, 1 );


						for (int batchIdx = currentBatchIdx; (batchIdx < start_Y_indexes_vec.size()) && (batchIdx < (currentBatchIdx + BATCH_SIZE)); batchIdx++) {	

							int start_Y_savedIdx = start_Y_indexes_vec[batchIdx];
							int start_Z_savedIdx = start_Z_indexes_vec[batchIdx];

                                                        int* C_cases_GPU_ptr = C_ptrGPU_cases + (batchIdx % BATCH_SIZE) * 2 * ((BLOCK_SIZE * SNP_CALC) * (BLOCK_SIZE * SNP_CALC));
                                                        int* C_controls_GPU_ptr = C_ptrGPU_controls + (batchIdx % BATCH_SIZE) * 2 * ((BLOCK_SIZE * SNP_CALC) * (BLOCK_SIZE * SNP_CALC));

							numCasesAfterPrunning_X_noPadding[0] = dimensionsPrunned_cases[0 * numSNPs + X_index];
							numCasesAfterPrunning_X_noPadding[1] = dimensionsPrunned_cases[1 * numSNPs + X_index];
							numControlsAfterPrunning_X_noPadding[0] = dimensionsPrunned_controls[0 * numSNPs + X_index];
							numControlsAfterPrunning_X_noPadding[1] = dimensionsPrunned_controls[1 * numSNPs + X_index];


							if((start_Z_savedIdx + BLOCK_SIZE) > numSNPs) {
								objectiveFunctionKernel<true><<<blocksPerGrid_objFun, workgroupSize_objFun, 0, cudaStreamforX>>>(C_cases_GPU_ptr, C_controls_GPU_ptr, d_output_individualSNP_popcountsForCases_filteredBy_X0_super1, d_output_individualSNP_popcountsForControls_filteredBy_X0_super1, d_output_individualSNP_popcountsForCases_filteredBy_X1_super1, d_output_individualSNP_popcountsForControls_filteredBy_X1_super1, d_output_individualSNP_popcountsForCases_filteredBy_X0_super2, d_output_individualSNP_popcountsForControls_filteredBy_X0_super2, d_output_individualSNP_popcountsForCases_filteredBy_X1_super2, d_output_individualSNP_popcountsForControls_filteredBy_X1_super2, d_tablePrecalc, d_output, d_output_packedIndices, start_Y_savedIdx, start_Z_savedIdx, X_index, d_output_pairwiseSNP_popcountsForCases, d_output_pairwiseSNP_popcountsForControls, numSNPs, numCases, numControls);	
							}
							else {
								objectiveFunctionKernel<false><<<blocksPerGrid_objFun, workgroupSize_objFun, 0, cudaStreamforX>>>(C_cases_GPU_ptr, C_controls_GPU_ptr, d_output_individualSNP_popcountsForCases_filteredBy_X0_super1, d_output_individualSNP_popcountsForControls_filteredBy_X0_super1, d_output_individualSNP_popcountsForCases_filteredBy_X1_super1, d_output_individualSNP_popcountsForControls_filteredBy_X1_super1, d_output_individualSNP_popcountsForCases_filteredBy_X0_super2, d_output_individualSNP_popcountsForControls_filteredBy_X0_super2, d_output_individualSNP_popcountsForCases_filteredBy_X1_super2, d_output_individualSNP_popcountsForControls_filteredBy_X1_super2, d_tablePrecalc, d_output, d_output_packedIndices, start_Y_savedIdx, start_Z_savedIdx, X_index, d_output_pairwiseSNP_popcountsForCases, d_output_pairwiseSNP_popcountsForControls, numSNPs, numCases, numControls);	
							}

						}

					}

					/* Waits for a given SNP X to be processed */
					cudaStreamSynchronize(cudaStreamforX);

					/* Updates indices of SNP X, and blocks of SNPs Y and Z */
					X_index += 1;
					start_Y = MAX(superblock_Y_start, floor(((double)X_index) / BLOCK_SIZE) * BLOCK_SIZE);	// avoids doing processing with blocks where none of the Y in block of SNPs are > SNP X
					start_Z = MAX(start_Y, superblock_Z_start);

				}

				/* Prepares for processing the next super-block Z and updates remaining indices */
				superblock_Z_start += SUPERBLOCK_SIZE;
				X_index = 0;
				start_Y = superblock_Y_start;
				start_Z = superblock_Z_start;

			}

			/* Prepares for processing the next super-block Y and updates remaining indices */
			superblock_Y_start += SUPERBLOCK_SIZE;
			superblock_Z_start = superblock_Y_start;
			X_index = 0;
			start_Y = superblock_Y_start;
			start_Z = superblock_Z_start;
		}

		clock_gettime(CLOCK_MONOTONIC, &t_iter_end);
		double timing_duration_iter = ((t_iter_end.tv_sec + ((double) t_iter_end.tv_nsec / 1000000000)) - (t_iter_start.tv_sec + ((double) t_iter_start.tv_nsec / 1000000000)));

		/* Prints time taken to execute a given unit of work */
		std::cout << "MPI rank " << mpiRank << ": "  << std::fixed << std::setprecision(3) << timing_duration_iter << " seconds" << std::endl;    


	}

	/* Copies best solution found from GPU memory to Host */
	cudaMemcpy(outputFromGpu, d_output, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(output_indexFromGpu_packedIndices, d_output_packedIndices, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	cudaFree(cases_A_ptrGPU);
	cudaFree(C_ptrGPU_cases);
	cudaFree(controls_A_ptrGPU);
	cudaFree(C_ptrGPU_controls);
	cudaFree(d_tablePrecalc);
	cudaFree(d_output);
	cudaFree(d_output_packedIndices);

	free(h_tablePrecalc);

	return cudaSuccess;
}

