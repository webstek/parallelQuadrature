#include<iostream>
// Cuda includes
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
// Local Header Files
#include<paratrap.h>
#include<serialtrap.h>

// TODO determine if there is bug in quadrature locations -> may be fixed 0,beta interval

// define dummy kernel 
__global__ void dummy(int n) {
	// thread number
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	i += n;
}

// Number of intervals to compute quadrature on, and number of threads per block
const int N = 10000000;
const int BLOCK_SIZE = 256;

int main(void) {
	// Call dummy kernel to initialize CUDA things
	int t = 1;
	dummy<<<1, BLOCK_SIZE >>> (t);

	std::cout << "Integral of exp(cos(x)) from 0 to 1 with " << N << " intervals...\n\n";

	// Call Parallel and Serial Quadrature schemes. Results in console.
	double para = parallelQuad(0, 1, N);
	double richardson = richardsonQuad(0, 1, N);
	double serial = serialQuad(0, 1, N);

	return 0;
}