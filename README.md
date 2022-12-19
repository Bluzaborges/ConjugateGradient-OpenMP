## Parallelized conjugate gradient method using OpenMP

A simple implementation of conjugate gradient method with the objective of parallelize using the OpenMP library in C.

The solution must be compiled with the following command:

```
gcc -o gradient gradient.c -fopenmp
```

And executed with:

```
./gradient <order of matrix> <number of threads>
```