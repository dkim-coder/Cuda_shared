#include "cuda_runtime.h"


#ifndef _matMul_cuh_
#define _matMul_cuh_

#define BLOCK_SIZE 5

typedef struct{
    int width;
    int height;
    int stride;
    double* elements;
}Matrix;

typedef struct {
    int width;
    int height;
    int depth;
    int stride_w;
    int stride_h;
    double* elements;
}Tensor3D;


void setMatrix(Matrix);
void matCPU(const Matrix, const Matrix, Matrix);
void printMatrix(const Matrix);


__device__ double GetElement(const Matrix, int, int);
__device__ void SetElement(Matrix, int, int, double);
__device__ Matrix GetSubMatrix(Matrix, int, int);
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);
void matGPU(const Matrix, const Matrix, Matrix);


int cublasMat(const Matrix, const Matrix, Matrix);

#endif 