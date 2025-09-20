# OpenCL, OpenMP, and MPI parallelisation

As part of the Advanced High Performance Computing unit at the University of Bristol, I parallelised a Lattice Boltzmann code using the OpenMP, and MPI libraries. All implementations are written in C. 

## About

High-performance D2Q9 Lattice Boltzmann Simulation implemented with MPI distributed memory parallelism. Achieved up to 32× speedup when scaling from 1 to 112 processes on the BCp4 cluster. Optimized Structure of Arrays (SoA) memory layout reduced cache misses and improved MPI transfer efficiency, sustaining ~191 GFLOPS/s throughput in large-scale runs. Includes roofline analysis confirming memory-bandwidth bound performance and detailed scalability benchmarks.

### OpenMP

``` shell
$ make clean
$ make openmp
$ ./d2q9-bgk input/input_128x128.params input/obstacles_128x128.dat
```

Or use the job script `qsub openmp_submit`

### MPI

``` shell
$ make clean
$ make mpi
$ mpirun -np 16 mpi_d2q9-bgk input/input_128x128.params input/obstacles_128x128.dat
```

Or use the job script `qsub mpi_submit`
