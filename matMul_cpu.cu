#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include "matMul.cuh"


void setMatrix(Matrix A) {
	srand((unsigned int)time(NULL));

	for (int i = 0; i < (A.height * A.width); i++) {
		A.elements[i] = (float)(rand() % 100) / 10;
	}
}


void matCPU(const Matrix A,const Matrix B, Matrix C) {
	int idx = 0;
	float tmp;

	for (int i = 0; i < A.height; i++) {
		for (int j = 0; j < B.width; j++) {
			tmp = 0;
			for (int l = 0; l < A.width; l++) {
				tmp += A.elements[i * A.width + l] * B.elements[l * B.width + j];
			}
			C.elements[idx] = tmp;
			idx++;
		}
	}
}

void printMatrix(const Matrix A) {
	printf("---------------------------------------------------------------------------------------------\n");
	int idx = 0;

	for (int i = 0; i < A.height; i++) {
		for (int j = 0; j < A.width; j++) {
			printf("%.8f ", A.elements[idx]);
			idx++;
		}
		printf("\n");
	}
}