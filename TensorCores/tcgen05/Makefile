OFFLINE ?= 0
# Default: PTX-only (driver JIT) to avoid ptxas rejecting tcgen05 on toolchains without tcgen05 support.
# Set OFFLINE=1 to request cubin (requires nvcc/ptxas that understands tcgen05, e.g., CUDA 12.8+ with PTX ISA 9.0).
ifeq ($(OFFLINE),1)
GENCODE_SM100 := -gencode=arch=compute_100,code=\"sm_100,compute_100\"
else
GENCODE_SM100 := -gencode=arch=compute_100,code=compute_100
$(info tcgen05: building PTX-only for sm_100 (driver JIT). Set OFFLINE=1 to emit cubin if your ptxas supports tcgen05.)
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
