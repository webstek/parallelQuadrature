#include<iostream>
#include<iomanip>
#include<math.h>
#include<chrono>

// CUDA includes
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


// GPU code for computing the function values on the grid points
__global__ void f_evaluate(double* f, double h, int n)
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


double richardsonQuad(const double alpha, const double beta, const int N, const int BLOCK_SIZE)
{
	std::cout << "Computing in parallel with Richardson Extrapolation\n";

	// Set timer start
	auto start = std::chrono::high_resolution_clock::now();

	// Compute interval size and half interval size
	const double h = (beta - alpha) / N;
	const double h_upon2 = 0.5 * h;

	// allocate host memory for function evaluations on h_upon2 sized grid
	const int ARRAY_BYTES = sizeof(double) * (2 * N + 1);
	double* h_farray = new double[2 * N + 1];

	// allocate gpu memory for function evaluations on grid
	double* d_farray;
	cudaMalloc(&d_farray, ARRAY_BYTES);
	cudaMemset(d_farray, 0, ARRAY_BYTES);

	// compute number of blocks required and launch kernal for each point in the grid
	int num_blocks = (2*N / BLOCK_SIZE) + 1;
	f_evaluate<<<num_blocks, BLOCK_SIZE >>>(d_farray, h_upon2, 2*N);

	// copy back the array from GPU memory, then free GPU memory
	cudaMemcpy(h_farray, d_farray, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	cudaFree(d_farray);

	// Compute approximations on both grid levels and remove memory allocation
	double trap_h_upon2 = 0.5 * (h_farray[0] + h_farray[2 * N]);
	double trap_h = trap_h_upon2;
	for (int k = 1; k < 2*N; k++) {
		trap_h_upon2 += h_farray[k];
		if (k % 2 == 0) {
			trap_h += h_farray[k];
		}
	}
	double intgrl = (4 / 3) * (h_upon2 * trap_h_upon2) - (1 / 3) * (h * trap_h);
	delete[] h_farray;

	// Set timer end, compute duration
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds> (end - start);

	// Print results to console
	std::cout << "Integral is approximately: " << std::fixed << std::setprecision(16) << intgrl << "\n";
	std::cout << "Computed in: " << duration.count() << " (microseconds)\n\n";

	return intgrl;
}