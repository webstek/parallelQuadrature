# Parallel Quadrature
Implementation and Testing of Parallel Quadrature Methods for Time-Save Potential

### Preliminary Results
- For accuracy up to 8 digits (N=~1 000 000), serial implementation is significantly faster (due to need to copy information from host to gpu)
- parallelQuad() time ~constant (170ms) for N in (10, 10 000 000)
  - That is, within variation between runs
- serialQuad() time linearly increases with N (10,10 000 000)
  - from ~10 μs to 100ms




For UBC Math 406 project
