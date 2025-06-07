blocksize ?= 512
superblock ?= 2048
batchsize ?= 32
numiter ?= 2048
tbsize ?= 256
snpsthread ?= 8

CXX = nvcc
CXXFLAGS = --default-stream per-thread -O3 -arch=sm_80 -lineinfo -Xcompiler -Icutlass-3.4.1/include -Icutlass-3.4.1/tools/util/include -Iinclude -DBLOCK_SIZE=$(blocksize) -DSUPERBLOCK_SIZE=$(superblock) -DBATCH_SIZE=$(batchsize) -DNUM_ITER=$(numiter) -DTB_SIZE=$(tbsize) -DSNPS_PER_THREAD=$(snpsthread) -g -lmpi	
EXE_NAME = epiclear
SOURCES = src/helper.cu src/tensorop.cu src/search.cu src/epiclear.cu
BINDIR = bin


epiclear:
	[ -d $(BINDIR) ] || mkdir $(BINDIR)
	$(CXX) $(SOURCES) $(CXXFLAGS) -o $(BINDIR)/$(EXE_NAME)


