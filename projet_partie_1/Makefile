CC=gcc -Wall -O3

OMP=-fopenmp

FILES= enhanced_algo utils #naive_algo
OBJECTS=$(addsuffix .o, $(FILES))
BIN=bin/enhanced_algo #bin/naive_algo

all : $(BIN)

$(BIN) : $(addprefix obj/, $(OBJECTS))
	$(CC) -o $@ $^ $(OMP) 

obj/%.o : src/%.c
	$(CC) -c -o $@ $^ $(OMP)

exec:
	$(BIN)

#.PHONY: clean mrproper

clean :
	rm -f bin/* obj/*

mrproper :
	rm -f bin/* obj/* saved/*
