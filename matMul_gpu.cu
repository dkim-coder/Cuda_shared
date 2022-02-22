#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "matMul.cuh"
#include <stdio.h>
#include <stdlib.h>


#define BLOCK_SIZE 16


__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}


__device__ void SetElement(Matrix A, int row, int col, float value)
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
    cudaMalloc(&d_A.elements, A.height * A.width * sizeof(float));
    cudaMemcpy(d_A.elements, A.elements, A.height * A.width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_B.elements, B.height * B.width * sizeof(float));
    cudaMemcpy(d_B.elements, B.elements, B.height * B.width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_C.elements, C.height * C.width * sizeof(float));

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((B.width + BLOCK_SIZE - 1) / BLOCK_SIZE, (A.height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    MatMulKernel << <dimGrid, dimBlock >> > (d_A, d_B, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU에서 행렬곱 실행시간 : % .3f\n", milliseconds);


    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, C.height * C.width * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;


    int x = BLOCK_SIZE * blockCol + col;
    int y = BLOCK_SIZE * blockRow + row;


    for (int k = 0; k < (A.width + BLOCK_SIZE - 1) / BLOCK_SIZE; k++) {

        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, k);

        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, k, blockCol);

        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        __syncthreads();

        if (k == ((A.width + BLOCK_SIZE - 1) / BLOCK_SIZE - 1)) {
            for (int e = 0; e < (A.width - BLOCK_SIZE * k); e++) {
                Cvalue += As[row][e] * Bs[e][col];
            }
        }
        else {
            for (int e = 0; e < BLOCK_SIZE; e++) {
                Cvalue += As[row][e] * Bs[e][col];
            }
        }

        __syncthreads();
    }

    if(x < B.width && y < A.height) SetElement(Csub, row, col, Cvalue);

}