# OpenCL, OpenMP, and MPI parallelisation

As part of the Advanced High Performance Computing unit at the University of Bristol, I parallelised a Lattice Boltzmann code using the OpenMP, and MPI libraries. All implementations are written in C. 

## About

The OpenMP, and MPI implementations are provided in the `opencl_d2q9-bgk.c`, `openmp_d2q9-bgk.c`, and `mpi_d2q9-bgk.c` files respectively. The kernels for the OpenCL implementation are provided in `kernels.cl`.

## Usage

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
