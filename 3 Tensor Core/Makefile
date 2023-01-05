tensor_core: mat_mult.cu
	nvcc -arch=sm_70 -o mat_mult.out mat_mult.cu
	./mat_mult.out 1001 1001 1001

clean:
	rm *.out