﻿#include<iostream>
#include<iomanip>
#include<math.h>
#include<chrono>
// CUDA includes
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


// GPU code for computing the function values on the grid points
__global__ void f_eval(double* f, double h, int n)
{
	// thread number
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i <= n) {
		// set x value for i
		double x = i * h;

		// compute the function value at x
		f[i] = exp(cos(x));
	}
}


// CPU Code to compute approximation of the integral
double parallelQuad(const double alpha, const double beta, const int N, const int BLOCK_SIZE)
{
	std::cout << "Computing integral in parallel\n";

	// Set timer start
	auto start = std::chrono::high_resolution_clock::now();

	// Compute interval size
	const double h = (beta - alpha) / N;

	// allocate host memory for function evaluations on grid
	const int ARRAY_BYTES = sizeof(double) * (N + 1);
	double * h_farray = new double [N + 1]; // limited to size < 1 million bc of stack size

	// allocate GPU memory for function evaluations on grid
	double* d_farray;
	cudaMalloc(&d_farray, ARRAY_BYTES);
	cudaMemset(d_farray, 0, ARRAY_BYTES);

	// compute number of blocks required and launch kernal for each point in the grid
	int num_blocks = (N / BLOCK_SIZE) + 1;
	f_eval <<<num_blocks, BLOCK_SIZE>>> (d_farray, h, N);

	// copy back the array from GPU memory, then free GPU memory
	cudaMemcpy(h_farray, d_farray, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	cudaFree(d_farray);

	// Compute approximation and remove memory allocation
	double intgrl = 0.5 * (h_farray[0] + h_farray[N]);
	for (int k = 1; k < N; k++) {
		intgrl += h_farray[k];
	}
	intgrl *= h;
	delete[] h_farray;

	// Set timer end, compute duration
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds> (end - start);


	// Print results to console
	std::cout << "Integral is approximately: " << std::fixed << std::setprecision(16) << intgrl << "\n";
	std::cout << "Computed in: " << duration.count() << " (microseconds)\n\n";

	// Return intgrl value
	return intgrl;
}