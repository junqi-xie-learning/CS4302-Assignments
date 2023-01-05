tensor_core: mat_mult.cu
	nvcc -arch=sm_70 -o mat_mult.out mat_mult.cu
	./mat_mult.out 1001 1001 1001

tensor_core_sim: mat_mult.cpp
	g++ -g -Wall -o mat_mult.out mat_mult.cpp tensor_core.cpp
	./mat_mult.out 1001 1001 1001

clean:
	rm *.out