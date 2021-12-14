#include<iostream>
#include<iomanip>
#include<chrono>
#include<math.h>


// Nonparallelized Trapezoidal rule quadrature for f(x) = exp(cos(x)) on alpha,beta with N intervals
double serialQuad(const double alpha, const double beta, const int N)
{

	std::cout << "Computing int_0^1 exp(cos(x)) in series with " << N+1 << " grid points...\n";

	// Set timer start
	auto start = std::chrono::high_resolution_clock::now();

	// Compute h
	double h = (beta - alpha) / N;

	// Compute approximation
	double intgrl = 0.5 * (exp(cos(alpha)) + exp(cos(beta)));
	for (int k = 1; k < N; k++) {
		intgrl += (exp(cos(h * k)));
	}
	intgrl *= h;

	// Set timer end, computer duration
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds > (end - start);

	// Print results to console
	std::cout << "Integral is approximately: " << std::fixed << std::setprecision(16) << intgrl << "\n";
	std::cout << "Computed in: " << duration.count() << " (milliseconds)\n\n";

	return intgrl;
}