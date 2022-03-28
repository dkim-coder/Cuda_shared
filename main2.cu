#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include"matMul.cuh"

#define _CTR_SEUCRE_NO_WARNINGS

/*
* C[a, b, c] += A[a, k] * B[k, c, b]
*/
void matGPU2(const Matrix A, const Tensor3D B, Tensor3D C);


int main() {
	int a, b, c, k;

    printf("input a value : ");
    scanf("%d", &a);
    printf("input b value : ");
    scanf("%d", &b);
    printf("input c value : ");
    scanf("%d", &c);
    printf("input k value : ");
    scanf("%d", &k);

    double** A;
    double*** B, *** C;
    
    
    // memory allocation
    A = (double**)malloc(sizeof(double*) * a);
    for (int i = 0; i < a; i++) {
        A[i] = (double*)malloc(sizeof(double) * k);
    }
    B = (double***)malloc(sizeof(double***) * k);
    for (int i = 0; i < k; i++) {
        B[i] = (double**)malloc(sizeof(double*) * c);
        for (int j = 0; j < c; j++) {
            B[i][j] = (double*)malloc(sizeof(double) * b);
        }
    }
    C = (double***)malloc(sizeof(double***) * a);
    for (int i = 0; i < a; i++) {
        C[i] = (double**)malloc(sizeof(double*) * b);
        for (int j = 0; j < b; j++) {
            C[i][j] = (double*)calloc(c, sizeof(double));   // set C --> 0
        }
    }
    
    
    // fill A, B
    srand((unsigned int)time(NULL));
    for (int i = 0; i < a; i++) {
        for (int j = 0; j < k; j++) {
            A[i][j] = (double)(rand() % 100) / 1000;
        }
    }
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < c; j++) {
            for (int l = 0; l < b; l++) {
                B[i][j][l] = (double)(rand() % 100) / 1000;
            }
        }
    }
    

    // C[a, b, c] += A[a, k] * B[k, c, b]
    clock_t start, end;

    // a,b,c,k
    start = clock();
    for (int i = 0; i < a; i++) {
        for (int j = 0; j < b; j++) {
            for (int l = 0; l < c; l++) {
                for (int m = 0; m < k; m++) {
                    C[i][j][l] += A[i][m] * B[m][l][j];
                }
            }
        }
    }
    end = clock();
    printf("a, b, c, k ---->>  CPU에서 행렬곱 실행시간 : % .8f second\n", (float)(end - start) / CLOCKS_PER_SEC);

    Matrix _A;
    Tensor3D _B, _C;
    _A.height = a; _A.width = _A.stride = k;
    _B.depth = k; _B.height = _B.stride_h = c; _B.width = _B.stride_w = b;
    _C.depth = a; _C.height = _C.stride_h = b; _C.width = _C.stride_w = c;
    
    // 메모리 할당
    _A.elements = (double*)malloc(sizeof(double) * a * k);
    _B.elements = (double*)malloc(sizeof(double) * k * c * b);
    _C.elements = (double*)malloc(sizeof(double) * a * b * c);
    // _A, _B 값 채우기
    for (int i = 0; i < a; i++) {
        for (int j = 0; j < k; j++) {
            _A.elements[k * i + j] = A[i][j];
        }
    }
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < c; j++) {
            for (int l = 0; l < b; l++) {
                _B.elements[c * b * i + b * j + l] = B[i][j][l];
            }
        }
    }   


    matGPU2(_A, _B, _C);

 
    // 값 비교
    for (int i = 0; i < a; i++) {
        for (int j = 0; j < b; j++) {
            for (int l = 0; l < c; l++) {
                if (C[i][j][l] != _C.elements[b * c * i + c * j + l]) printf("오류\n");
            }
        }
    }


    // free memory
    for (int i = 0; i < a; i++) {
        free(A[i]);
    }
    free(A);
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < c; j++) {
            free(B[i][j]);
        }
    }
    for (int i = 0; i < k; i++) {
        free(B[i]);
    }
    free(B);
    for (int i = 0; i < a; i++) {
        for (int j = 0; j < b; j++) {
            free(C[i][j]);
        }
    }
    for (int i = 0; i < a; i++) {
        free(C[i]);
    }
    free(C);

    free(_A.elements);
    free(_B.elements);
    free(_C.elements);

	return EXIT_SUCCESS;
}



__global__ void MatMulKernel2(const Matrix A, const Tensor3D B, Tensor3D C) {
    int depth = blockIdx.z * blockDim.z + threadIdx.z;    //a
    int row = blockIdx.y * blockDim.y + threadIdx.y;  //b
    int col = blockIdx.x * blockDim.x + threadIdx.x;  //c

    double Cvalue = 0;
    for (int e = 0; e < A.width; e++) {
        Cvalue += A.elements[A.width * depth + e] * B.elements[B.width * B.height * e + B.width * col + row];
    }
    
    if (depth < C.depth && row < C.height && col < C.width) {
        C.elements[C.width * C.height * depth + C.width * row + col] = Cvalue;
    }

    return;
}



__device__ Tensor3D GetSubTensor3D(const Tensor3D A, const int blockDepth, const int blockRow, const int blockCol) {
    Tensor3D subT;
    subT.depth = BLOCK_SIZE;
    subT.height = BLOCK_SIZE;
    subT.width = BLOCK_SIZE;
    subT.stride_h = A.height;
    subT.stride_w = A.width;
    subT.elements = &A.elements[A.width * A.height * BLOCK_SIZE * blockDepth + A.width * BLOCK_SIZE * blockRow + BLOCK_SIZE * blockCol];

    return subT;
}

__device__ void SetSubTensor3D(const Tensor3D A, const int depth, const int row, const int col,const double value) {
    A.elements[A.stride_h * A.stride_w * depth + A.stride_w * row + col] = value;
}

__device__ double GetElementTensor3D(const Tensor3D A, const int depth, const int row, const int col) {
    return A.elements[A.stride_h * A.stride_w * depth + A.stride_w * row + col];
}

// shared memory 사용하기 미완성
__global__ void MatMulKernel3(const Matrix A, const Tensor3D B, Tensor3D C) {
   
}




void matGPU2(const Matrix A, const Tensor3D B, Tensor3D C) {
    Matrix d_A;
    Tensor3D d_B, d_C;

    d_A.height = A.height; d_A.width = d_A.stride = A.width;
    d_B.depth = B.depth; d_B.height = d_B.stride_h = B.height; d_B.width = d_B.stride_w = B.width;
    d_C.depth = C.depth; d_C.height = d_C.stride_h = C.height; d_C.width = d_C.stride_w = C.width;

    //메모리 할당, 복사
    cudaMalloc(&d_A.elements, sizeof(double) * d_A.height * d_A.width);
    cudaMemcpy(d_A.elements, A.elements, sizeof(double) * d_A.height * d_A.width, cudaMemcpyHostToDevice);
    cudaMalloc(&d_B.elements, sizeof(double) * d_B.depth * d_B.height * d_B.width);
    cudaMemcpy(d_B.elements, B.elements, sizeof(double) * d_B.depth * d_B.height * d_B.width, cudaMemcpyHostToDevice);
    cudaMalloc(&d_C.elements, sizeof(double) * d_C.depth * d_C.height * d_C.width);
    cudaMemset(d_C.elements, 0, sizeof(double) * d_C.depth * d_C.height * d_C.width);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((d_C.width + BLOCK_SIZE - 1) / BLOCK_SIZE, (d_C.height + BLOCK_SIZE - 1) / BLOCK_SIZE, (d_C.depth + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    MatMulKernel2 << <dimGrid, dimBlock >> > (d_A, d_B, d_C);
    //MatMulKernel3 << <dimGrid, dimBlock >> > (d_A, d_B, d_C);

    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU에서 행렬, 텐서곱 실행시간 : % .8f second\n", milliseconds / 1000.);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(C.elements, d_C.elements, sizeof(double) * d_C.depth * d_C.height * d_C.width, cudaMemcpyDeviceToHost);


    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);

    return;
}