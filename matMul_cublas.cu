#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "matMul.cuh"

// column major order
#define IDX2C(i,j,ld) (((j)*(ld))+(i))    // i == row, j == column


int cublasMat(const Matrix A, const Matrix B, Matrix C) {
    cudaError_t cudaStat;   // cuda_runtime library status
    cublasStatus_t stat;    // cudlas_v2 library status
    cublasHandle_t handle;  

    Matrix d_A, d_B, d_C;
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    size_t size_a = d_A.height * d_A.width * sizeof(double);
    size_t size_b = d_B.height * d_B.width * sizeof(double);
    size_t size_c = d_C.height * d_C.width * sizeof(double);

    int m, n, k;
    m = d_A.height;
    n = d_B.width;
    k = d_A.width;
    
    // memory allocation
    cudaStat = cudaMalloc(&d_A.elements, size_a);
    if (cudaStat != cudaSuccess) {
        printf("device memory allocation failed");
        return EXIT_FAILURE;
    }
    cudaStat = cudaMalloc(&d_B.elements, size_b);
    if (cudaStat != cudaSuccess) {
        printf("device memory allocation failed");
        return EXIT_FAILURE;
    }
    cudaStat = cudaMalloc(&d_C.elements, size_c);
    if (cudaStat != cudaSuccess) {
        printf("device memory allocation failed");
        return EXIT_FAILURE;
    }


    // Create handle
    stat = cublasCreate_v2(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS initialization failed\n");
        cudaFree(d_A.elements);
        cudaFree(d_B.elements);
        cudaFree(d_C.elements);

        return EXIT_FAILURE;
    }

 
    // set Matrix d_A, d_B
    stat = cublasSetMatrix(A.height, A.width, sizeof(double), A.elements, A.height, d_A.elements, d_A.height);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("data download from host to device failed");
        cublasDestroy_v2(handle);
        cudaFree(d_A.elements);
        cudaFree(d_B.elements);
        cudaFree(d_C.elements);
        
        return EXIT_FAILURE;
    }
    stat = cublasSetMatrix(B.height, B.width, sizeof(double), B.elements, B.height, d_B.elements, d_B.height);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("data download from host to device failed");
        cublasDestroy_v2(handle);
        cudaFree(d_A.elements);
        cudaFree(d_B.elements);
        cudaFree(d_C.elements);
        
        return EXIT_FAILURE;
    }
    
    // d_C = d_A * d_B
    double const alpha(1.0);
    double const beta(0.0);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // ???
    stat = cublasDgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B.elements, n, d_A.elements, k, &beta, d_C.elements, n);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("mutiply matrix in device failed");
        cudaFree(d_A.elements);
        cudaFree(d_B.elements);
        cudaFree(d_C.elements);
        cublasDestroy_v2(handle);
        
        return EXIT_FAILURE;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("cublas --->>> GPU에서 행렬곱 실행시간 : % .8f second\n", milliseconds / 1000.);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    
   // Get matrix
    stat = cublasGetMatrix(d_C.height, d_C.width, sizeof(double), d_C.elements, d_C.height, C.elements, C.height);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("data download from device to host failed");
        cudaFree(d_A.elements);
        cudaFree(d_B.elements);
        cudaFree(d_C.elements);
        cublasDestroy_v2(handle);

        return EXIT_FAILURE;
    }


    // Free memory on GPU side
    cublasDestroy_v2(handle);
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);

    return EXIT_SUCCESS;
}