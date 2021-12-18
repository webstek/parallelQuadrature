#include<iostream>
#include<fstream>
#include<iomanip>
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
int N[8] = { 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000 };
const int BLOCK_SIZE = 256;

int main(void) {
	// Call dummy kernel to initialize CUDA things
	int t = 1;
	dummy<<<1, BLOCK_SIZE >>> (t);

	std::cout << "Integral of sqrt((exp(cos(pow(pow(pow(x, x), x), x))))); from 0 to 1 with " << N << " intervals...\n\n";

	std::ofstream myfile;
	myfile.open("results_time_f2.txt", std::ofstream::trunc);
	myfile.close();
	myfile.open("results_intgrl_f2.txt");

	for (int i = 0; i < 8; i++) {

		std::cout << "N = " << N[i] << "\n";

		// Call Parallel and Serial Quadrature schemes. Results in console.
		double para = parallelQuad(0, 1, N[i]);
		double richardson = richardsonQuad(0, 1, N[i]);
		double serial = serialQuad(0, 1, N[i]);

		myfile << std::fixed << std::setprecision(16) << para << "," << richardson << "," << serial << "\n";
	}
	myfile.close();
	return 0;
}