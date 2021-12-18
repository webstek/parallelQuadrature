#include<iostream>
#include<iomanip>
#include<fstream>
#include<chrono>
#include<math.h>

// Function to compute integral from 0 to 1 of.
double F(double x) {
	return sqrt((exp(cos(pow(pow(pow(x, x), x), x)))));
}

// Nonparallelized Trapezoidal rule quadrature for f(x) = exp(cos(x)) on alpha,beta with N intervals
double serialQuad(const double alpha, const double beta, const int N)
{

	std::cout << "Computing integral in series\n";
	std::ofstream myfile;
	myfile.open("results_time_f2.txt", std::ios_base::app);

	// Set timer start
	auto start = std::chrono::high_resolution_clock::now();

	// Compute h
	double h = (beta - alpha) / N;

	// Compute approximation
	double intgrl = 0.5 * F(alpha) + F(beta);
	for (int k = 1; k < N; k++) {
		double x = h * k;
		intgrl += F(x);
	}
	intgrl *= h;

	// Set timer end, computer duration, write to file
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds > (end - start);
	myfile << duration.count() << ",\n";
	myfile.close();

	// Print results to console
	std::cout << "Integral is approximately: " << std::fixed << std::setprecision(16) << intgrl << "\n";
	std::cout << "Computed in: " << duration.count() << " (microseconds)\n\n";

	return intgrl;
}