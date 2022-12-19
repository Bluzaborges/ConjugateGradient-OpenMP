#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define IMAX 100
#define ERROR 0.00000001

float **create_matrix_A(int order){
	
	int i;
	float **A = (float**)calloc(order, order * sizeof(float *));
	for (i = 0; i < order; i++)
		A[i] = (float *)calloc(order, order * sizeof(float));	
	
	for (i = 0; i < order; i++){
		A[i][i] = 4;
		
		if (i + 1 < order)
			A[i + 1][i] = A[i][i + 1] = 1;
	}
	
	return A;
}

float *create_array_b(int size){

	float *b = (float *)malloc(size * sizeof(float));
	
	for (int i = 0; i < size; i++)
		b[i] = 1;
		
	return b;
}

float *create_array_x(int size){
	return 	(float *)calloc(size, size * sizeof(float));
}

void matrix_by_array_product_openmp(float **matrix, float *array, float *product, int order, int id, int numberOfThreads){
	for (int i = id; i < order; i += numberOfThreads){
		product[i] = 0;
		for (int j = 0; j < order; j++)
			product[i] += array[j] * matrix[i][j];
	}
}

void array_subtraction(float *array1, float *array2, float *difference, int size){
	for (int i = 0; i < size; i++)
		difference[i] = array1[i] - array2[i];
}

void array_sum(float *array1, float *array2, float *sum, int size){
	for (int i = 0; i < size; i++)
		sum[i] = array1[i] + array2[i];
}

float scalar_array_product(float *array1, float *array2, int size){

	float product = 0;
	
	for (int i = 0; i < size; i++)
		product += array1[i] * array2[i];
		
	return product;
}

void scalar_by_array_product(float scalar, float *array, float *product, int size){
	for (int i = 0; i < size; i++)
		product[i] = array[i] * scalar;
}

int conjugate_gradient(float **A, float *b, float *x, int order, int numberOfThreads, double *time){
	
	int i = 0, id;
	float *vaux = (float *)malloc(order * sizeof(float));
	float *q = (float *)malloc(order * sizeof(float));
	float *d = (float *)malloc(order * sizeof(float));
	float *r = (float *)malloc(order * sizeof(float));
	float alpha, newSigma, oldSigma, beta;
	
	omp_set_num_threads(numberOfThreads);
	
	double startTime = omp_get_wtime();
	
	#pragma omp parallel private(id)
	{
		id = omp_get_thread_num();
	
		matrix_by_array_product_openmp(A, x, vaux, order, id, numberOfThreads);
		
		#pragma omp barrier
		
		if (id == 0){
			array_subtraction(b, vaux, r, order);
			
			memcpy(d, r, order * sizeof(float));
			
			newSigma = scalar_array_product(r, r, order);
		}
		
		#pragma omp barrier
		
		while (i < IMAX && newSigma > ERROR){
			
			matrix_by_array_product_openmp(A, d, q, order, id, numberOfThreads);
			
			#pragma omp barrier
			
			if (id == 0){
				alpha = newSigma / scalar_array_product(d, q, order);
				
				scalar_by_array_product(alpha, d, vaux, order);
				
				array_sum(x, vaux, x, order);
				
				scalar_by_array_product(alpha, q, vaux, order);
				
				array_subtraction(r, vaux, r, order);
				
				oldSigma = newSigma;
				
				newSigma = scalar_array_product(r, r, order);
				
				beta = newSigma / oldSigma;
				
				scalar_by_array_product(beta, d, vaux, order);
				
				array_sum(r, vaux, d, order);
				
				i++;
			}
			
			#pragma omp barrier
		}
	}
	
	double finalTime = omp_get_wtime();
	
	*time = finalTime - startTime;

	free(vaux);
	free(q);
	free(d);
	free(r);
	
	return i;
}

int main(int argc, char **argv){

	if (argc != 3){
		printf("%s <order of matrix> <number of threads>\n", argv[0]);
		exit(0);
	}
	
	int order = atol(argv[1]), iterations;
	int numberOfThreads = atoi(argv[2]);
	
	double time;
	
	float **A = create_matrix_A(order);
	float *b = create_array_b(order);
	float *x = create_array_x(order);
	
	iterations = conjugate_gradient(A, b, x, order, numberOfThreads, &time);
	
	printf("Time %.4f | Number of iterations: %d\n", time, iterations);
}