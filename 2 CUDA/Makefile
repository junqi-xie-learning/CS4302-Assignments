mat_mult: mat_mult.cu
	nvcc -o mat_mult.out mat_mult.cu
	./mat_mult.out 1001 1001 1001

rbf: rbf.cu
	nvcc -o rbf.out rbf.cu
	./rbf.out 2 32 10000 10000

clean:
	rm *.out
