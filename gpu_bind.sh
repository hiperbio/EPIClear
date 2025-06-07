#!/bin/bash

# mpi_rank=$PMI_RANK		# Intel MPI	
mpi_rank=$OMPI_COMM_WORLD_RANK	# OpenMPI

# Modify if the targeted system has a different number of GPUs.
export CUDA_VISIBLE_DEVICES=$((mpi_rank % 4))

$@
