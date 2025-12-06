GENCODE_SM100 := -gencode=arch=compute_100,code=\"sm_100,compute_100\"

CUOPTS = $(GENCODE_SM100)

CC := nvcc

SRC_LDST = tmem_ldst.cu
EXE_LDST = tmem_ldst

SRC_CP_SHIFT = tmem_cp_shift.cu
EXE_CP_SHIFT = tmem_cp_shift

SRC_MMA_WS = mma_ws_tmem.cu
EXE_MMA_WS = mma_ws_tmem

all: ldst cp_shift mma_ws

ldst:
	$(CC) $(CUOPTS) $(SRC_LDST) -o $(EXE_LDST)

cp_shift:
	$(CC) $(CUOPTS) $(SRC_CP_SHIFT) -o $(EXE_CP_SHIFT)

mma_ws:
	$(CC) $(CUOPTS) $(SRC_MMA_WS) -o $(EXE_MMA_WS)

clean:
	rm -f $(EXE_LDST) $(EXE_CP_SHIFT) $(EXE_MMA_WS) *.o
