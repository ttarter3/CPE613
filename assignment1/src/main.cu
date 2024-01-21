
// Project Headers
#include "TarterHelper.c"

// 3rd Party Headers


// Standard Libraries
#include <iostream>
#include <cstdlib>

template <typename Datatype>
__host__ __device__ void axpy(const int N, const Datatype alpha, const Datatype *X, const int incX, Datatype *Y, const int incY) {
	// Generalized implmentation of axpy
	Y[incY] = alpha * X[incX] + Y[incY];
}

template <typename Datatype>
__global__ void axpy(const int N, const Datatype alpha, const Datatype *X, Datatype *Y) {
	// Generic Grid Stide Loop - Useful for iterating over all the data requested
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < N; 
         i += blockDim.x * gridDim.x) 
      {
          axpy(N, alpha, X, i, Y, i);
      }
}

int main() {
    // Size of the vectors
    const int size = 1000;

    // Host vectors
    float* h_x = new float[size];
    float* h_y = new float[size];
	float* h_y_orig = new float[size];
	float alpha = 2;


    // Initialize host vectors
    for (int i = 0; i < size; ++i) {
        h_x[i] = std::rand();
        h_y[i] = std::rand();
		h_y_orig[i] = h_y[i];
		
    }

    // Device vectors
    float* d_x, *d_y;
    cudaMalloc((void**)&d_x, size * sizeof(float));
    cudaMalloc((void**)&d_y, size * sizeof(float));

    // Copy host vectors to device
    cudaMemcpy(d_x, h_x, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    axpy<<<blocksPerGrid, threadsPerBlock>>>(size, alpha, d_x, d_y);

	checkKernelError();

    // Copy result back to host
    cudaMemcpy(h_y, d_y, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Print some results
    std::cout << "axpy test:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << alpha << "*" << h_x[i] << " + " << h_y_orig[i] << " = " << h_y[i] << std::endl;
    }

    // Clean up
    delete[] h_x;
    delete[] h_y;
	delete[] h_y_orig;
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}

