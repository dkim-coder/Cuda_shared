#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "matMul.cuh"
#include <stdio.h>
#include <stdlib.h>


__device__ double GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}


__device__ void SetElement(Matrix A, int row, int col, double value)
{
    A.elements[row * A.stride + col] = value;
}


__device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
    Matrix Asub;
    Asub.width = BLOCK_SIZE;
    Asub.height = BLOCK_SIZE;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    
    return Asub;
}


void matGPU(const Matrix A, const Matrix B, Matrix C)
{
    Matrix d_A, d_B, d_C;
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    d_C.width = d_C.stride = C.width; d_C.height = C.height;

    size_t size_a = d_A.height * d_A.width * sizeof(double);
    size_t size_b = d_B.height * d_B.width * sizeof(double);
    size_t size_c = d_C.height * d_C.width * sizeof(double);
    
    // memory allocation
    cudaMalloc(&d_A.elements, size_a);
    cudaMemcpy(d_A.elements, A.elements, size_a, cudaMemcpyHostToDevice);
    cudaMalloc(&d_B.elements, size_b);
    cudaMemcpy(d_B.elements, B.elements, size_b, cudaMemcpyHostToDevice);
    cudaMalloc(&d_C.elements, size_c);
    cudaMemset(d_C.elements, 0, size_c);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((B.width + dimBlock.x - 1) / dimBlock.x, (A.height + dimBlock.y - 1) / dimBlock.y);
    

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    MatMulKernel <<<dimGrid, dimBlock >>> (d_A, d_B, d_C);    

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU에서 행렬곱 실행시간 : % .8f second\n", milliseconds/1000.);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size_c, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(const Matrix A, const Matrix B, Matrix C)
{
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int row = threadIdx.y;
    int col = threadIdx.x;
    int x = BLOCK_SIZE * blockCol + col;
    int y = BLOCK_SIZE * blockRow + row;

    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];

    double Cvalue = 0.0;

    for (int k = 0; k < (A.width + BLOCK_SIZE - 1) / BLOCK_SIZE; k++) {
        Matrix Asub = GetSubMatrix(A, blockRow, k);
        Matrix Bsub = GetSubMatrix(B, k, blockCol);

        As[row][col] = 0.0;
        Bs[row][col] = 0.0;

        if (k == ((A.width + BLOCK_SIZE - 1) / BLOCK_SIZE - 1)) {
            if (y < A.height && (col < A.width - k * BLOCK_SIZE)) As[row][col] = GetElement(Asub, row, col);
            if (x < B.width && (row < A.width - k * BLOCK_SIZE)) Bs[row][col] = GetElement(Bsub, row, col);
        }
        else {
            if (y < A.height) As[row][col] = GetElement(Asub, row, col);
            if (x < B.width) Bs[row][col] = GetElement(Bsub, row, col);
        }
        __syncthreads();
        
        for (int e = 0; e < BLOCK_SIZE; e++) {
            Cvalue += As[row][e] * Bs[e][col];
        }

        __syncthreads();
    }
    
    if (x < B.width && y < A.height) {
        SetElement(Csub, row, col, Cvalue);
    }
}