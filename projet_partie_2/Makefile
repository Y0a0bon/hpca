CC=gcc -Wall -O3

NVCC=nvcc

LIB=-lm -L/usr/local/cuda/lib64/ -lcuda -lcudart
OMP=-fopenmp

## Debugging
DEBUGGING = y

ifeq ($(DEBUGGING), y)
 CFLAGS=-g 
 CUDA_FLAGS = --generate-code arch=compute_35,code=sm_35 -g -G #-DCUDA_DEBUG 
else
 CFLAGS=-O3
 CUDA_FLAGS = --generate-code arch=compute_35,code=sm_35 -O3 #-DCUDA_DEBUG 
endif 

FILES=dpr linear_algo dpr_cuda

all : $(FILES)

################################

dpr :
	echo 'export OMP_NUM_THREADS=8'
	$(CC) -c -o obj/dpr.o src/dpr.c $(OMP)
	$(CC) -c -o obj/utils.o src/utils.c $(OMP)
	$(CC) -o bin/dpr obj/dpr.o obj/utils.o $(OMP)

linear_algo :
	$(CC) -c -o obj/utils.o src/utils.c
	$(CC) -c -o obj/stack.o src/stack.c
	$(CC) -c -o obj/linear_algo.o src/linear_algo.c
	$(CC) -o bin/linear_algo obj/linear_algo.o obj/utils.o obj/stack.o

dpr_cuda :
	$(NVCC) -c -o obj/dpr_cuda.o src/dpr_cuda.cu $(CUDA_FLAGS)
	$(NVCC) -c -o obj/utils.o src/utils.cu $(CUDA_FLAGS)
	$(CC) -o bin/dpr_cuda obj/dpr_cuda.o obj/utils.o $(LIB)

################################

exec :
	./bin/dpr ./data/ex_N100000_res100000
	./bin/linear_algo ./data/ex_N100000_res100000
	./bin/dpr_cuda ./data/ex_N100000_res100000

#.PHONY: clean mrproper

clean :
	rm -f bin/* obj/*
