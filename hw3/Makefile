
#########################################################################
#
# Makefile for Assignment 3
#
#########################################################################

NVCC=nvcc
NVCCFLAG =-c -rdc=true

user_program.o: user_program.cu
	$(NVCC) $(NVCCFLAG) user_program.cu
virtual_memory.o: virtual_memory.cu
	$(NVCC) $(NVCCFLAG) virtual_memory.cu
main.o: main.cu
	$(NVCC) $(NVCCFLAG) main.cu
all: user_program.o virtual_memory.o main.o
	$(NVCC) -arch=sm_37 -std=c++11 -cudart=shared -rdc=true user_program.o virtual_memory.o main.o -o test
clean:
	rm -rf test *.o