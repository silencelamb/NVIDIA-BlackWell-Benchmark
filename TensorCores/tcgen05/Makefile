GENCODE_SM100 := -gencode=arch=compute_100,code=\"sm_100,compute_100\"

CUDA_MAJ := $(shell nvcc --version 2>/dev/null | grep release | sed 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1/')
CUDA_MIN := $(shell nvcc --version 2>/dev/null | grep release | sed 's/.*release \([0-9]*\)\.\([0-9]*\).*/\2/')
CUDA_VER_OK := $(shell if [ -n "$(CUDA_MAJ)" ] && { [ $(CUDA_MAJ) -gt 12 ] || { [ $(CUDA_MAJ) -eq 12 ] && [ $(CUDA_MIN) -ge 8 ]; }; }; then echo yes; else echo no; fi)

ifeq ($(CUDA_VER_OK),no)
$(error tcgen05 requires CUDA >= 12.8 (PTX ISA 9.0) for tcgen05 instructions; detected $(CUDA_MAJ).$(CUDA_MIN))
endif

CUOPTS = $(GENCODE_SM100)

CC := nvcc

SRC_LDST = tmem_ldst.cu
EXE_LDST = tmem_ldst

SRC_CP_SHIFT = tmem_cp_shift.cu
EXE_CP_SHIFT = tmem_cp_shift

SRC_MMA_WS = mma_ws_tmem.cu
EXE_MMA_WS = mma_ws_tmem

MMA_KIND ?= f16

all: ldst cp_shift mma_ws

ldst:
	$(CC) $(CUOPTS) $(SRC_LDST) -o $(EXE_LDST)

cp_shift:
	$(CC) $(CUOPTS) $(SRC_CP_SHIFT) -o $(EXE_CP_SHIFT)

mma_ws:
	$(CC) $(CUOPTS) $(SRC_MMA_WS) -o $(EXE_MMA_WS) -DMMA_KIND=$(MMA_KIND)

clean:
	rm -f $(EXE_LDST) $(EXE_CP_SHIFT) $(EXE_MMA_WS) *.o
