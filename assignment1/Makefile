

.PHONY: clean run

run: 
	@./.runJob.sh

clean:
	rm -rf main *.qsub_out

main: ./src/main.cu
	nvcc -o main -I./include -I/home/uahclsd0009/workspace/CPE613/cuda-samples/Common ./src/main.cu
	