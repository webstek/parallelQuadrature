#include<iostream>
#include<iomanip>
#include <math.h>
#include<chrono>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Computes the integral: int_0^1{ exp(cos(x)) dx} using a uniform grid compound Trapezoidal parallel method
#define BLOCK_SIZE 256

// Endpoints and number of grid intervals
const double ALPHA = 0;
const double BETA = 1;
const int N = 1000;
const double h = (BETA - ALPHA) / N;

void print_array(double* array, int size);

// GPU code for computing the function values on the grid points
__global__ void double_f_eval(double* f, double h, int n)
{
	// thread number
	// TODO may need to change below formula to incorporate more threads than in 1 thread block
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i <= n) {
		// set x value for i
		double x = i * h;

		// compute the function value at x (for compount Trapezoidal rule) --> the function is f(x) = exp(cos(x)) in this case.
		f[i] = exp(cos(x));
	}
}


// CPU Code to compute approximation of the integral
int main(int argc, char** argv)
{
	std::cout << "Computing int_0^1 exp(cos(x)) in parallel on " << N+1 << " grid points...\n";

	// Set timer start
	auto start = std::chrono::high_resolution_clock::now();

	// allocate host memory for function evaluations on grid
	const int ARRAY_BYTES = sizeof(double) * (N + 1);
	double h_farray[N + 1]; // limited to size < 1 million bc of stack size
	// double *h_farray = (double*)malloc(ARRAY_BYTES);

	// allocate GPU memory for function evaluations on grid
	double* d_farray;
	cudaMalloc(&d_farray, ARRAY_BYTES);
	cudaMemset(d_farray, 0, ARRAY_BYTES);

	// compute number of blocks required and launch kernal for each point in the grid
	int num_blocks = (N / BLOCK_SIZE) + 1;
	double_f_eval <<<num_blocks, BLOCK_SIZE>>> (d_farray, h, N);

	// copy back the array from GPU memory, then free GPU memory
	cudaMemcpy(h_farray, d_farray, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	cudaFree(d_farray);

	// Compute approximation
	double intgrl = 0.5 * (h_farray[0] + h_farray[N]);
	for (int k = 1; k < N; k++) {
		intgrl += h_farray[k];
	}
	intgrl *= h;


	// Set timer end, computer duration
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds> (end - start);


	// Print results to console
	std::cout << "Integral is approximately: " << std::fixed << std::setprecision(20) << intgrl << "\n";
	std::cout << "Computed in: " << duration.count() << " (milliseconds)\n";

	return 0;
}

// Display's an Array
void print_array(double* array, int size)
{
	std::cout << "[";
	for (int i = 0; i < size-1; i++) { std::cout << array[i] <<", "; }
	std::cout << array[size-1] << "]\n";
}