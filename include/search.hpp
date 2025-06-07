#ifndef SEARCH_H_   
#define SEARCH_H_

/* Two out of the three base genotypes are represented with two bits per {SNP, sample} tuple. */
#define SNP_CALC 2

/* Interation order (k=3). */
#define INTER_OR 3

/* There are 3^3 (=27) possible third-order genotypes. */
#define SNP_COMB 27		

/* Matrix operations are used to count the occurence rate of 2^3 (= 8) out of 27 genotypes, being the remaining 19 analytically derived. */
#define SNP_COMB_CALC 8

/* The controls and cases matrices are expected to be padded to multiples of PADDING_SAMPLES in regard to samples. */
#define PADDING_SAMPLES 1024	

/* Used in the GPU kernel that scores SNP combinations based on the corresponding contingency tables. */
#define BLOCK_OBJFUN	4	

/* Search function executed by each MPI worker. */
cudaError_t EpistasisDetectionSearch(unsigned int* datasetCases_host_matrixA, unsigned int* datasetControls_host_matrixA, int numSNPs, int numCases, int numControls, uint numSNPsWithPadding, int numCasesWithPadding, int numControlsWithPadding, float * outputFromGpu, unsigned long long int * output_indexFromGpu_packedIndices, int mpiRank);

#endif
