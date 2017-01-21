CC=gcc -Wall -O3
NVCC=nvcc

LIB=-lm -L/usr/local/cuda/lib64/ -lcuda -lcudart
OMP= -fopenmp

## Debugging
DEBUGGING = y

ifeq ($(DEBUGGING), y)
 CFLAGS=-g 
 CUDA_FLAGS = --generate-code arch=compute_35,code=sm_35 -g -G #-DCUDA_DEBUG 
else
 CFLAGS=-O3
 CUDA_FLAGS = --generate-code arch=compute_35,code=sm_35 -O3 #-DCUDA_DEBUG 
endif 

FILES = naive_algo enhanced_algo \
	naive_algo_cuda naive_algo_opt_cuda \
	enhanced_algo_cuda enhanced_algo_opt_cuda \
	utils

all : $(FILES)

################################

naive_algo :
	echo 'export OMP_NUM_THREADS=16'
	$(CC) -c -o obj/naive_algo.o src/naive_algo.c $(OMP)
	$(CC) -c -o obj/utils.o src/utils.c $(OMP)
	$(CC) -o bin/naive_algo obj/naive_algo.o obj/utils.o $(OMP)

enhanced_algo :
	echo 'export OMP_NUM_THREADS=16'
	$(CC) -c -o obj/enhanced_algo.o src/enhanced_algo.c $(OMP)
	$(CC) -c -o obj/utils.o src/utils.c $(OMP)
	$(CC) -o bin/enhanced_algo obj/enhanced_algo.o obj/utils.o $(OMP)

################################

naive_algo_cuda :
	$(NVCC) -c -o obj/naive_algo_cuda.o src/naive_algo_cuda.cu $(CUDA_FLAGS)
	$(NVCC) -c -o obj/utils.o src/utils.cu $(CUDA_FLAGS)
	$(CC) -o bin/naive_algo_cuda obj/naive_algo_cuda.o obj/utils.o $(LIB)

naive_algo_opt_cuda :
	$(NVCC) -c -o obj/naive_algo_opt_cuda.o src/naive_algo_opt_cuda.cu $(CUDA_FLAGS)
	$(NVCC) -c -o obj/utils.o src/utils.cu $(CUDA_FLAGS)
	$(CC) -o bin/naive_algo_opt_cuda obj/naive_algo_opt_cuda.o obj/utils.o $(LIB)

################################

enhanced_algo_cuda :
	$(NVCC) -c -o obj/enhanced_algo_cuda.o src/enhanced_algo_cuda.cu $(CUDA_FLAGS)
	$(NVCC) -c -o obj/utils.o src/utils.cu $(CUDA_FLAGS)
	$(CC) -o bin/enhanced_algo_cuda obj/enhanced_algo_cuda.o obj/utils.o $(LIB)

enhanced_algo_opt_cuda :
	$(NVCC) -c -o obj/enhanced_algo_opt_cuda.o src/enhanced_algo_opt_cuda.cu $(CUDA_FLAGS)
	$(NVCC) -c -o obj/utils.o src/utils.cu $(CUDA_FLAGS)
	$(CC) -o bin/enhanced_algo_opt_cuda obj/enhanced_algo_opt_cuda.o obj/utils.o $(LIB)

################################

exec :
	./bin/naive_algo ./data/ex_N500_res7854
	./bin/enhanced_algo ./data/ex_N100000_res100000
	./bin/naive_algo_cuda ./data/ex_N100000_res100000
	./bin/enhanced_algo_cuda ./data/ex_N100000_res100000

#.PHONY: clean

clean :
	rm -f bin/* obj/*

