GENCODE_SM90 := -gencode=arch=compute_90,code=\"sm_90,compute_90\"
GENCODE_SM100 := -gencode=arch=compute_100,code=\"sm_100,compute_100\"

CUOPTS = $(GENCODE_SM90) $(GENCODE_SM100)

CC := nvcc

INCLUDE :=
LIB :=

SRC_LAT = remote_latency.cu
EXE_LAT = remote_latency

SRC_BW = remote_bw.cu
EXE_BW = remote_bw

all: latency bandwidth

latency:
	$(CC) $(CUOPTS) $(SRC_LAT) -o $(EXE_LAT) -I$(INCLUDE) -L$(LIB) -lcudart

bandwidth:
	$(CC) $(CUOPTS) $(SRC_BW) -o $(EXE_BW) -I$(INCLUDE) -L$(LIB) -lcudart

clean:
	rm -f $(EXE_LAT) $(EXE_BW) *.o

run:
	./$(EXE_LAT)

profile:
	nvprof ./$(EXE_LAT)
