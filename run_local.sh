#!/bin/bash

NUMPROC=$(($1 + 1))
mpirun -np $NUMPROC --map-by :OVERSUBSCRIBE ./gpu_bind.sh ./bin/epiclear $2 
