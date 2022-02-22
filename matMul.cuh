#include "cuda_runtime.h"


#ifndef _matMul_cuh_
#define _matMul_cuh_

typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;


void setMatrix(Matrix);
void matCPU(const Matrix, const Matrix, Matrix);
void printMatrix(const Matrix);


__device__ float GetElement(const Matrix, int, int);
__device__ void SetElement(Matrix, int, int, float);
__device__ Matrix GetSubMatrix(Matrix, int, int);
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);
void matGPU(const Matrix, const Matrix, Matrix);

#endif 