
#########################################################################
#
# Makefile for Assignment 4
#
#########################################################################

NVCC=nvcc
NVCCFLAG =-c -rdc=true 

user_program.o: user_program.cu
	$(NVCC) $(NVCCFLAG) user_program.cu
file_system.o: file_system.cu
	$(NVCC) $(NVCCFLAG) file_system.cu
main.o: main.cu
	$(NVCC) $(NVCCFLAG) main.cu
all: user_program.o file_system.o main.o
	$(NVCC) -arch=sm_37 -std=c++11 -cudart=shared -rdc=true user_program.o file_system.o main.o -o test
clean:
	rm -rf test *.o