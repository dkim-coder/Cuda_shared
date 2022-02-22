#include "cuda_runtime.h"


#ifndef _matMul_cuh_
#define _matMul_cuh_

typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;


void setMatrix(float*, const int);
void matCPU(const float*, const float*, float*, const int, const int, const int);
void printCPU(const float*, const int, const int);


__device__ float GetElement(const Matrix, int, int);
__device__ void SetElement(Matrix, int, int, float);
__device__ Matrix GetSubMatrix(Matrix, int, int);
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);
void matGPU(const Matrix, const Matrix, Matrix);
void printGPU(Matrix);

#endif 