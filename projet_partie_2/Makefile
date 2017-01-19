CC=gcc -Wall -O3

OMP=-fopenmp

FILES= 	dpr \
	linear_algo \
#utils
#OBJECTS=$(addsuffix .o, $(FILES))
#BIN=bin/dpr #bin/enhanced_algo #bin/naive_algo

#all : $(FILES)
#
#$(BIN) : $(addprefix obj/, $(OBJECTS))
#	$(CC) -o $@ $^ $(OMP)
#
#obj/%.o : src/%.c
#	$(CC) -c -o $@ $^ $(OMP)
#
#exec:
#	$(BIN)

all : $(FILES)

################################

dpr :
	$(CC) -c -o obj/dpr.o src/dpr.c $(OMP)
	$(CC) -c -o obj/utils.o src/utils.c $(OMP)
	$(CC) -o bin/dpr obj/dpr.o obj/utils.o $(OMP)

linear_algo :
	$(CC) -c -o obj/utils.o src/utils.c
	$(CC) -c -o obj/stack.o src/stack.c
	$(CC) -c -o obj/linear_algo.o src/linear_algo.c
	$(CC) -o bin/linear_algo obj/linear_algo.o obj/utils.o obj/stack.o

################################

#.PHONY: clean mrproper

clean :
	rm -f bin/* obj/*

