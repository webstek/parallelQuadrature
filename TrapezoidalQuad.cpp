#include<iostream>

// Local Header Files
#include<paratrap.h>
#include<serialtrap.h>

// Number of intervals to compute quadrature on
const int N = 10000;

int main() {
	// Call Parallel and Serial Quadrature schemes here
	double para_integral = parallelQuad(0,1,N);
	double serial_integral = serialQuad(0,1,N);

	return 0;
}