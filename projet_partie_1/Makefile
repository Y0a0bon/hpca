CC=gcc -Wall -O3
NVCC=nvcc

LIB=-lm -L/usr/local/cuda/lib64/ -lcuda -lcudart

## Debugging
DEBUGGING = y

ifeq ($(DEBUGGING), y)
 CFLAGS=-g 
 CUDA_FLAGS = --generate-code arch=compute_35,code=sm_35 -g -G #-DCUDA_DEBUG 
else
 CFLAGS=-O3
 CUDA_FLAGS = --generate-code arch=compute_35,code=sm_35 -O3 #-DCUDA_DEBUG 
endif 

FILES= naive_algo_cuda utils
OBJECTS=$(addsuffix .o, $(FILES))
BIN=bin/naive_algo_cuda

all : $(BIN)

$(BIN) : $(addprefix obj/, $(OBJECTS))
	$(CC) -o $@ $^ $(LIB)

obj/%.o : src/%.cu
	$(NVCC) -c -o $@ $^ $(CUDA_FLAGS)

exec:
	$(BIN)

#.PHONY: clean mrproper

clean :
	rm -f bin/* obj/*

mrproper :
	rm -f bin/* obj/* saved/*
