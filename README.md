# EPIClear

EPIClear is a tool implementing a state-of-the-art algorithm for performing exhaustive high-order epistasis detection searches with binarized processing on GPU tensor cores.
It accelerates third-order searches through a technique that leverages genotypic information to reduce the volume of AND+POPC matrix operations performed to count genotypes.

## Setup

### Requirements

* CUDA Toolkit (tested with 12.2)
* CUTLASS (tested with 3.4.1)
* GPU with AND+POPC on tensor cores

### Compilation example

Compile the EPIClear binary (`epiclear`) with the following command to specialize the super-block size to 8192 SNPs:

```bash
$ make superblock=8192
```

The number of SNPs per block (`blocksize`), the number of SNPs per super-block (`superblock`), the batch size used in batched AND+POPC matrix operations (`batchsize`), the number of evaluation rounds processed per unit of work (`numiter`), the thread block size used in the kernel performing data compaction (`tbsize`) and the number of SNPs it processes per thread (`snpsthread`) can be specialized to better suit the characteristics of a particular dataset and the targeted system, which can be a computer with a single GPU or a GPU-accelerated cluster.

## Usage example

Run the following command to use a local system with 4 GPUs to process an example dataset (download from <a href="https://drive.google.com/file/d/1UNs0yCFuRiXVQG6cEN7sjvzhyVV438k7/view?usp=sharing">here</a>): 

```bash
$ sh run_local.sh 4 datasets/db_8192snps_524288samples_10_20_70.txt   
```

The use of a GPU-accelerated computer cluster (e.g. the GPU partition of a supercomputer) can be accomplished as exemplified in `run_slurm.sh`, adapting the script to reflect the desired node count and the particular characteristics of the system.
