/**
 *
 * epiclear.cu: third-order epistasis detection searches using AND+POPC operations that leverages genotypic information to improve time-to-solution
 *
 * High-Performance Computing Architectures and Systems (HPCAS) Group, INESC-ID

 * Contact: Ricardo Nobre <ricardo.nobre@inesc-id.pt>
 *
 */

#include <iostream>
#include <iomanip>      
#include <sstream>
#include <vector>
#include <cfloat>
#include <string>
#include <libgen.h>
#include <mpi.h>

#include "helper.hpp"
#include "reduction.hpp"
#include "search.hpp"

int main(int argc, const char *arg[]) {

	/* Get CUDA device properties for device with id equal to 0. */
	cudaDeviceProp device_properties;
	cudaError_t result = cudaGetDeviceProperties(&device_properties, 0);

	if (result != cudaSuccess) {
		std::cerr << "Could not get device properties: " << cudaGetErrorString(result) << std::endl;
		return 1;
	}

	if ((device_properties.major * 10 +  device_properties.minor) < 80) {
		std::cerr << "Compute capability 8.0 (or above) is required." << std::endl;
		return 1;
	}


	if(argc < 2) {
		std::cerr << "Usage: epiclear dataset.txt" << std::endl;
		return 1;
	}


	/* Reads information about input dataset. */

	/* File with information and pointers to dataset. */
	FILE* fStream = fopen(arg[1], "r");     		
	if(fStream == NULL) {
		std::cerr << "File '" << arg[1] << "' does not exist!" << std::endl;
		return 1;
	}

	char* ts = strdup(arg[1]);
	char* pathToDataset = dirname(ts);	

	/* First line represents the number of SNPs. */
	char line[MAX_CHAR_ARRAY];	
	char* ret = fgets(line, MAX_CHAR_ARRAY, fStream); 	
	uint numSNPs = atoi(line);

	/* Second line represents the filename with controls data. */
	char controlsFileName[MAX_CHAR_ARRAY];
	ret = fgets(controlsFileName, MAX_CHAR_ARRAY, fStream);	
	
	/* Removes trailing newline character. */
	controlsFileName[strcspn(controlsFileName, "\n")] = 0;	

	/* Third line represents the number of controls. */
	ret = fgets(line, MAX_CHAR_ARRAY, fStream); 	
	uint numControls = atoi(line);

	/* Forth line represents the filename with cases data. */
	char casesFileName[MAX_CHAR_ARRAY];
	ret = fgets(casesFileName, MAX_CHAR_ARRAY, fStream);
	
	/* Removes trailing newline character. */	
	casesFileName[strcspn(casesFileName, "\n")] = 0;	

	/* Fifth line represents the number of cases. */
	ret = fgets(line, MAX_CHAR_ARRAY, fStream); 
	uint numCases = atoi(line);


	/* Calculates number of distinct blocks and padds number of SNPs to process to the block size. */
	uint numBlocks = ceil((float)numSNPs / (float)BLOCK_SIZE);
	uint numSNPsWithPadding = numBlocks * BLOCK_SIZE;

	/* Padds the number of controls and of cases. */
	uint numCasesWithPadding = ceil((float)numCases / PADDING_SAMPLES) * PADDING_SAMPLES;	
	uint numControlsWithPadding = ceil((float)numControls / PADDING_SAMPLES) * PADDING_SAMPLES;


	/* Allocates pinned memory for holding controls and cases dataset matrices.
	   Each 32-bit 'unsigned int' holds 32 binary values representing genotype information.
	 */

	int numSamplesCases_32packed = ceil(((float) numCasesWithPadding) / 32.0f);
	int numSamplesControls_32packed = ceil(((float) numControlsWithPadding) / 32.0f);

	int datasetCases_32packed_size = numSamplesCases_32packed * numSNPsWithPadding * SNP_CALC;
	unsigned int* datasetCases_32packed_matrixA = NULL;
	result = cudaHostAlloc((void**)&datasetCases_32packed_matrixA, datasetCases_32packed_size * sizeof(unsigned int), cudaHostAllocDefault );     
	if(datasetCases_32packed_matrixA == NULL) {
		std::cerr << "Problem allocating Host memory for cases" << std::endl;
	}

	int datasetControls_32packed_size = numSamplesControls_32packed * numSNPsWithPadding * SNP_CALC;    
	unsigned int* datasetControls_32packed_matrixA = NULL;
	result = cudaHostAlloc((void**)&datasetControls_32packed_matrixA, datasetControls_32packed_size * sizeof(unsigned int), cudaHostAllocDefault );
	if(datasetControls_32packed_matrixA == NULL) {
		std::cerr << "Problem allocating Host memory for controls" << std::endl;
	}


	/* Reads dataset (controls and cases data) from storage device.
	   Input dataset must be padded with zeros in regard to cases and controls. */

	size_t numElem;
	std::string absolutePathToCasesFile = std::string(pathToDataset) + "/" + casesFileName;
	FILE *ifp_cases = fopen(absolutePathToCasesFile.c_str(), "rb");
	numElem = fread(datasetCases_32packed_matrixA, sizeof(unsigned int), numSamplesCases_32packed * numSNPs * SNP_CALC, ifp_cases);
	if(numElem != datasetCases_32packed_size) {
		std::cerr << "Problem loading cases from storage device" << std::endl;
	}
	fclose(ifp_cases);

	std::string absolutePathToControlsFile = std::string(pathToDataset) + "/" + controlsFileName;
	FILE *ifp_controls = fopen(absolutePathToControlsFile.c_str(), "rb");
	numElem = fread(datasetControls_32packed_matrixA, sizeof(unsigned int), numSamplesControls_32packed * numSNPs * SNP_CALC, ifp_controls);
	if(numElem != datasetControls_32packed_size) {
		std::cerr << "Problem loading controls from storage device" << std::endl;
	}
	fclose(ifp_controls);


	/* Initializes the MPI environment */
	MPI_Init(NULL, NULL);

	/* Gets the number of processes */
	int mpi_world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

	/* Gets the rank of the process */
	int mpi_world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_world_rank);

	/* Gets the name of the processor */
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len);

	/* Prints a hello world message */
	printf("Hello world from %s, rank %d out of %d\n", processor_name, mpi_world_rank, mpi_world_size);

        /* Waits for all MPI processes */
        MPI_Barrier(MPI_COMM_WORLD);

	if(mpi_world_rank == 0) {
		printf("Using %d MPI processes.\n", mpi_world_size);

		/* Prints information about dataset and number of distinct blocks of SNPs to process. */
		std::cout << "Num. SNPs: " << numSNPs << std::endl;
		std::cout << "Num. Blocks of SNPs: " << numBlocks << std::endl;
		std::cout << "Num. Cases: " << numCases << std::endl;
		std::cout << "Num. Controls: " << numControls << std::endl;
	}

	/* Waits for all MPI processes */
	MPI_Barrier(MPI_COMM_WORLD); 


	/* Starts measuring time */
	struct timespec t_start, t_end;
	clock_gettime(CLOCK_MONOTONIC, &t_start);


	/* Indices of a block of SNPs Y, an SNP X and a block of SNPs X.
	   Initialized to 0 to represent combining the first SNP (0) twice with the first block of SNPs (starting SNP is 0). */
	int start_Y = 0;
	int X_index = 0;
	int start_Z = 0;


	/* The amount of MPI workers is the world size minus one, since rank 0 is the coordinator.
	   Used to keep track of the active workers. */
	int numActiveSlaves = mpi_world_size - 1;	

	/* Stores an association score and the associated set of SNP indices.
	   Used to keep track of the output of executing the epistasis detection search portion mapped to the GPU interfaced through an MPI worker. */
	float outputFromGpu = FLT_MAX;
	unsigned long long int output_indexFromGpu_packedIndices;

	/* Stores three numbers in a 64-bit variable representing the starting SNP of a block of SNPs Y, an SNP X and the starting SNP of a block of SNPs Z.
	   Determines which SNP combinations are evaluated by an MPI worker as part of a given unit of work. */
	unsigned long long int start_Y_and_X_index_and_start_Z;

	/* Indexes representing the first SNPs of the superblocks Y and Z.
	   Used as part of generalization of  */
	unsigned int superblock_Y_start = 0;
	unsigned int superblock_Z_start = 0;



	/* Counter used to space wout the starting evaluation round of subsequent units of work. 
	   Starts at NUM_ITER, so that it adds the first set of {X, Block Y, Block Z} */
	int iterAcc = NUM_ITER;


	/* Implements the MPI rank 0 (coordinator) logic.
	   Sends to MPI workers (all other MPI processes) information on the initial evaluation round to process as part of a given unit of work. */
	if (mpi_world_rank == 0) {

		MPI_Status mpiStatus;

		/* Rank 0 keeps its role as the coordinator as long as there are active MPI workers. */
		while ( numActiveSlaves > 0 ) {

			if(iterAcc >= NUM_ITER) {	

				iterAcc = 0;

				MPI_Recv(NULL, 0, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &mpiStatus);
				int slaveRank = mpiStatus.MPI_SOURCE;

				/* Keeps sending more workload to MPI workers while there are SNP combinations to process */
				if(superblock_Y_start < numSNPs) {

					start_Y_and_X_index_and_start_Z = (((unsigned long long int) start_Y) << 0) | (((unsigned long long int) X_index) << 20) | (((unsigned long long int) start_Z) << 40);
					MPI_Send(&start_Y_and_X_index_and_start_Z, 1, MPI_UNSIGNED_LONG_LONG, slaveRank, 0, MPI_COMM_WORLD);

				}
				else {	/* Instructs MPI worker to stop if there is no more work to do. */

					printf("Stopping worker with MPI rank %d\n", slaveRank);
					unsigned long long stopSlave = 0xffffffffffffffff;	
					MPI_Send (&stopSlave, 1, MPI_UNSIGNED_LONG_LONG, slaveRank , 0, MPI_COMM_WORLD);

					numActiveSlaves--;
				}

			}

			/* Increments starting SNP of block Z in a way that is determined by the block size parameter. */			
			start_Z += BLOCK_SIZE;

			/* Assesses if all blocks of super-block Z have already been combined with a given block Y.
			   If that is the case, increments starting SNP of block Y in a way that is determined by the block size parameter, and updates pointer to SNPs Z */
			if(start_Z >= (superblock_Z_start + SUPERBLOCK_SIZE)) {     

				start_Y += BLOCK_SIZE;
				start_Z = MAX(start_Y, superblock_Z_start); /* Takes into account that super-block Y and super-block Z might point to the same range of SNPs */	
			}	

			if((start_Y >= (superblock_Y_start + SUPERBLOCK_SIZE))) {
				X_index += 1;
				start_Y = MAX(superblock_Y_start, floor(((double)X_index) / BLOCK_SIZE) * BLOCK_SIZE);	/* Avoids processing with blocks Y where none of the SNPs has a higher index than SNP X */
				start_Z = MAX(start_Y, superblock_Z_start);
			}

			if(X_index >= (superblock_Y_start + SUPERBLOCK_SIZE)) {
				superblock_Z_start += SUPERBLOCK_SIZE;
				X_index = 0;
				start_Y = superblock_Y_start;
				start_Z = superblock_Z_start;				
			}

			if(superblock_Z_start >= numSNPsWithPadding) {
				superblock_Y_start += SUPERBLOCK_SIZE;
				superblock_Z_start = superblock_Y_start;
				X_index = 0;
				start_Y = superblock_Y_start;
				start_Z = superblock_Z_start;
			}

			iterAcc += 1;
		}


	} else {


		/* Launches the epistasis detection search. */

		result = EpistasisDetectionSearch(
				datasetCases_32packed_matrixA,		/* Cases matrix. */
				datasetControls_32packed_matrixA,	/* Controls matrix. */
				numSNPs,                                /* Number of SNPs. */
				numCases,                               /* Number of cases. */
				numControls,                            /* Number of controls. */
				numSNPsWithPadding,                     /* Number of SNPs padded to block size. */
				numCasesWithPadding,     		/* Number of cases padded to PADDING_SIZE. */
				numControlsWithPadding,     		/* Number of controls padded to PADDING_SIZE. */
				&outputFromGpu,				/* Score of best score found. */
				&output_indexFromGpu_packedIndices,	/* Indexes of SNPs of set that results in best score. */
				mpi_world_rank
				);

		if(result != cudaSuccess) {
			std::cerr << "Epistasis detection search failed." << std::endl;
		}

	}


	/* Reduces the scores taking into account the local optima found on the different MPI processes and identifies the SNP triplet associated with the overall best score */ 

	float * outputFromGpuPerMpiProcess; 
	unsigned long long int * output_indexFromGpu_packedIndicesPerMpiProcess;

	if ( mpi_world_rank == 0) { 
		outputFromGpuPerMpiProcess = (float *)malloc(mpi_world_size*1*sizeof(float)); 
		output_indexFromGpu_packedIndicesPerMpiProcess = (unsigned long long int *)malloc(mpi_world_size*1*sizeof(unsigned long long int));
	} 

	MPI_Gather( &outputFromGpu, 1, MPI_FLOAT, outputFromGpuPerMpiProcess, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);	
	MPI_Gather( &output_indexFromGpu_packedIndices, 1, MPI_UNSIGNED_LONG_LONG, output_indexFromGpu_packedIndicesPerMpiProcess, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);        

	float outputFromGpuAllMpiProcesses = FLT_MAX;
	unsigned long long output_indexFromGpu_packedIndicesAllMpiProcesses;

	if(mpi_world_rank == 0) {
		for(int i = 0; i < mpi_world_size; i++) {
			if(outputFromGpuPerMpiProcess[i] < outputFromGpuAllMpiProcesses) {
				outputFromGpuAllMpiProcesses = outputFromGpuPerMpiProcess[i];
				output_indexFromGpu_packedIndicesAllMpiProcesses = output_indexFromGpu_packedIndicesPerMpiProcess[i];
			}
		}

	}


	/* Waits for all MPI processes to finish */
	MPI_Barrier(MPI_COMM_WORLD);

	/* Gets ending time stamp */
	clock_gettime(CLOCK_MONOTONIC, &t_end);

	/* Signals no more MPI calls are performed */
	MPI_Finalize();


	/* MPI coordinator (rank 0) prints information related to the search,
	   including runtime, performance scaled to sample size and the number of unique combinations evaluated. */
	if(mpi_world_rank == 0) {

		double timing_duration_mpi = ((t_end.tv_sec + ((double) t_end.tv_nsec / 1000000000)) - (t_start.tv_sec + ((double) t_start.tv_nsec / 1000000000)));

		std::cout << "-------------------------------" << std::endl << "{SNP_X_i, SNP_Y_i, SNP_Z_i}: SCORE\t->\t{" << ((output_indexFromGpu_packedIndicesAllMpiProcesses >> 0) & 0x1FFFFF) << ", " << ((output_indexFromGpu_packedIndicesAllMpiProcesses >> 21) & 0x1FFFFF) << ", " << ((output_indexFromGpu_packedIndicesAllMpiProcesses >> 42) & 0x1FFFFF) << "}: " << std::fixed << std::setprecision(6) << outputFromGpuAllMpiProcesses << std::endl;

		unsigned long long numCombinations = n_choose_k(numSNPs, INTER_OR);

		std::cout << "Wall-clock time:\t" << std::fixed << std::setprecision(3) << timing_duration_mpi << " seconds" << std::endl;    

		std::cout << "Num. unique sets per sec. (scaled to sample size): " << std::fixed << std::setprecision(3) << (((double) numCombinations * (double) (numCases + numControls) / (double)(timing_duration_mpi)) / 1e12) << " x 10^12" << std::endl;

		std::cout << "Unique sets of SNPs evaluated (k=" << INTER_OR << "): " << numCombinations << std::endl;
	}

	return result == cudaSuccess ? 0 : 1;	

}

