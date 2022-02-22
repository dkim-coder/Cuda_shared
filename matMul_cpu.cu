#include<stdio.h>
#include<stdlib.h>
#include<time.h>


void setMatrix(float* A, const int size) {
	srand((unsigned int)time(NULL));

	for (int i = 0; i < size; i++) {
		A[i] = (float)(rand() % 100) / 10;
	}
}


void matCPU(const float* A, const float* B, float* C, const int m, const int n, const int k) {
	int idx = 0;
	float tmp;

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			tmp = 0;
			for (int l = 0; l < k; l++) {
				tmp += A[i * k + l] * B[l * n + j];
			}
			C[idx] = tmp;
			idx++;
		}
	}
}

void printCPU(const float* A, const int height, const int width) {
	printf("matrix multiplication on CPU\n");
	printf("---------------------------------------------------------------\n");
	int idx = 0;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			printf("%.3f ", A[idx]);
			idx++;
		}
		printf("\n");
	}
}