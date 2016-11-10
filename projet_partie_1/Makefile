CXX=gcc -Wall -O3

OMP=#-fopenmp

FILES= enhanced_algo #utils #naive_algo
OBJECTS=$(addsuffix .o, $(FILES))
BIN=bin/enhanced_algo

all : $(BIN)

$(BIN) : $(addprefix obj/, $(OBJECTS))
	$(CXX) -o $@ $^ $(LDFLAGS) $(OMP) 

obj/%.o : src/%.c
	$(CXX) -c -o $@ $^ $(OMP)

exec:
	$(BIN)

clean :
	rm -f bin/* obj/*
