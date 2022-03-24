#include "matMul.cuh"
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define _CTR_SEUCRE_NO_WARNINGS

typedef struct {
    int width;
    int height;
    int depth;
    int stride_w;
    int stride_h;
    double* elements;
}Matrix3D;

/*
* C[a, b, c] += A[a, k] * B[k, c, b]
*/

void zeroMat(double*** M, const int a, const int b, const int c);   // C 0으로 초기화
void matGPU2(const Matrix3D A, const Matrix3D B, Matrix3D C);


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
            C[i][j] = (double*)calloc(c, sizeof(double));
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
    /*
    zeroMat(C, a, b, c);

    // a,b,k,c
    start = clock();
    for (int i = 0; i < a; i++) {
        for (int j = 0; j < b; j++) {
            for (int l = 0; l < k; l++) {
                for (int m = 0; m < c; m++) {
                    C[i][j][m] += A[i][m] * B[l][m][j];
                }
            }
        }
    }
    end = clock();
    printf("a, b, k, c ---->>  CPU에서 행렬곱 실행시간 : % .8f second\n", (float)(end - start) / CLOCKS_PER_SEC);
    zeroMat(C, a, b, c);

    // a,c,b,k
    start = clock();
    for (int i = 0; i < a; i++) {
        for (int j = 0; j < c; j++) {
            for (int l = 0; l < b; l++) {
                for (int m = 0; m < k; m++) {
                    C[i][l][j] += A[i][m] * B[m][j][l];
                }
            }
        }
    }
    end = clock();
    printf("a, c, b, k ---->>  CPU에서 행렬곱 실행시간 : % .8f second\n", (float)(end - start) / CLOCKS_PER_SEC);
    zeroMat(C, a, b, c);

    // a,c,k,b
    start = clock();
    for (int i = 0; i < a; i++) {
        for (int j = 0; j < c; j++) {
            for (int l = 0; l < k; l++) {
                for (int m = 0; m < b; m++) {
                    C[i][m][j] += A[i][l] * B[l][j][m];
                }
            }
        }
    }
    end = clock();
    printf("a, c, k, b ---->>  CPU에서 행렬곱 실행시간 : % .8f second\n", (float)(end - start) / CLOCKS_PER_SEC);
    zeroMat(C, a, b, c);

    // a,k,b,c
    start = clock();
    for (int i = 0; i < a; i++) {
        for (int j = 0; j < k; j++) {
            for (int l = 0; l < b; l++) {
                for (int m = 0; m < c; m++) {
                    C[i][l][m] += A[i][j] * B[j][m][l];
                }
            }
        }
    }
    end = clock();
    printf("a, k, b, c ---->>  CPU에서 행렬곱 실행시간 : % .8f second\n", (float)(end - start) / CLOCKS_PER_SEC);
    zeroMat(C, a, b, c);
    
    // a,k,c,b
    start = clock();
    for (int i = 0; i < a; i++) {
        for (int j = 0; j < k; j++) {
            for (int l = 0; l < c; l++) {
                for (int m = 0; m < b; m++) {
                    C[i][m][l] += A[i][j] * B[j][l][m];
                }
            }
        }
    }
    end = clock();
    printf("a, k, c, b ---->>  CPU에서 행렬곱 실행시간 : % .8f second\n", (float)(end - start) / CLOCKS_PER_SEC);
    zeroMat(C, a, b, c);

    // b,a,c,k
    start = clock();
    for (int i = 0; i < b; i++) {
        for (int j = 0; j < a; j++) {
            for (int l = 0; l < c; l++) {
                for (int m = 0; m < k; m++) {
                    C[j][i][l] += A[j][m] * B[m][l][i];
                }
            }
        }
    }
    end = clock();
    printf("b, a, c, k ---->>  CPU에서 행렬곱 실행시간 : % .8f second\n", (float)(end - start) / CLOCKS_PER_SEC);
    zeroMat(C, a, b, c);

    // b,a,k,c
    start = clock();
    for (int i = 0; i < b; i++) {
        for (int j = 0; j < a; j++) {
            for (int l = 0; l < k; l++) {
                for (int m = 0; m < c; m++) {
                    C[j][i][m] += A[j][l] * B[l][m][i];
                }
            }
        }
    }
    end = clock();
    printf("b, a, k, c ---->>  CPU에서 행렬곱 실행시간 : % .8f second\n", (float)(end - start) / CLOCKS_PER_SEC);
    zeroMat(C, a, b, c);
    //b,c,a,k
    start = clock();
    for (int i = 0; i < b; i++) {
        for (int j = 0; j < c; j++) {
            for (int l = 0; l < a; l++) {
                for (int m = 0; m < k; m++) {
                    C[l][i][j] += A[l][m] * B[m][j][i];
                }
            }
        }
    }
    end = clock();
    printf("b, c, a, k ---->>  CPU에서 행렬곱 실행시간 : % .8f second\n", (float)(end - start) / CLOCKS_PER_SEC);
    zeroMat(C, a, b, c);
    //b,c,k,a
    start = clock();
    for (int i = 0; i < b; i++) {
        for (int j = 0; j < c; j++) {
            for (int l = 0; l < k; l++) {
                for (int m = 0; m < a; m++) {
                    C[m][i][j] += A[m][l] * B[l][j][i];
                }
            }
        }
    }
    end = clock();
    printf("b, c, k, a ---->>  CPU에서 행렬곱 실행시간 : % .8f second\n", (float)(end - start) / CLOCKS_PER_SEC);
    zeroMat(C, a, b, c);
    //b,k,a,c
    start = clock();
    for (int i = 0; i < b; i++) {
        for (int j = 0; j < k; j++) {
            for (int l = 0; l < a; l++) {
                for (int m = 0; m < c; m++) {
                    C[l][i][m] += A[l][j] * B[j][m][i];
                }
            }
        }
    }
    end = clock();
    printf("b, k, a, c ---->>  CPU에서 행렬곱 실행시간 : % .8f second\n", (float)(end - start) / CLOCKS_PER_SEC);
    zeroMat(C, a, b, c);
    //b,k,c,a
    start = clock();
    for (int i = 0; i < b; i++) {
        for (int j = 0; j < k; j++) {
            for (int l = 0; l < c; l++) {
                for (int m = 0; m < a; m++) {
                    C[m][i][l] += A[m][j] * B[j][l][i];
                }
            }
        }
    }
    end = clock();
    printf("b, k, c, a ---->>  CPU에서 행렬곱 실행시간 : % .8f second\n", (float)(end - start) / CLOCKS_PER_SEC);
    zeroMat(C, a, b, c);
    //c,a,b,k
    start = clock();
    for (int i = 0; i <c; i++) {
        for (int j = 0; j < a; j++) {
            for (int l = 0; l < b; l++) {
                for (int m = 0; m < k; m++) {
                    C[j][l][i] += A[j][m] * B[m][i][l];
                }
            }
        }
    }
    end = clock();
    printf("c, a, b, k ---->>  CPU에서 행렬곱 실행시간 : % .8f second\n", (float)(end - start) / CLOCKS_PER_SEC);
    zeroMat(C, a, b, c);
    //c,a,k,b
    start = clock();
    for (int i = 0; i < c; i++) {
        for (int j = 0; j < a; j++) {
            for (int l = 0; l < k; l++) {
                for (int m = 0; m < b; m++) {
                    C[j][m][i] += A[j][l] * B[l][i][m];
                }
            }
        }
    }
    end = clock();
    printf("c, a, k, b ---->>  CPU에서 행렬곱 실행시간 : % .8f second\n", (float)(end - start) / CLOCKS_PER_SEC);
    zeroMat(C, a, b, c);
    //c,b,a,k
    start = clock();
    for (int i = 0; i < c; i++) {
        for (int j = 0; j < b; j++) {
            for (int l = 0; l < a; l++) {
                for (int m = 0; m < k; m++) {
                    C[l][j][i] += A[l][m] * B[m][i][j];
                }
            }
        }
    }
    end = clock();
    printf("c, b, a, k ---->>  CPU에서 행렬곱 실행시간 : % .8f second\n", (float)(end - start) / CLOCKS_PER_SEC);
    zeroMat(C, a, b, c);
    //c,b,k,a
    start = clock();
    for (int i = 0; i < c; i++) {
        for (int j = 0; j < b; j++) {
            for (int l = 0; l < k; l++) {
                for (int m = 0; m < a; m++) {
                    C[m][j][i] += A[m][l] * B[l][m][j];
                }
            }
        }
    }
    end = clock();
    printf("c, b, k, a ---->>  CPU에서 행렬곱 실행시간 : % .8f second\n", (float)(end - start) / CLOCKS_PER_SEC);
    zeroMat(C, a, b, c);
    //c,k,a,b
    start = clock();
    for (int i = 0; i < c; i++) {
        for (int j = 0; j < k; j++) {
            for (int l = 0; l < a; l++) {
                for (int m = 0; m < b; m++) {
                    C[l][m][i] += A[l][j] * B[j][i][m];
                }
            }
        }
    }
    end = clock();
    printf("c, k, a, b ---->>  CPU에서 행렬곱 실행시간 : % .8f second\n", (float)(end - start) / CLOCKS_PER_SEC);
    zeroMat(C, a, b, c);
    //c,k,b,a
    start = clock();
    for (int i = 0; i < c; i++) {
        for (int j = 0; j < k; j++) {
            for (int l = 0; l < b; l++) {
                for (int m = 0; m < a; m++) {
                    C[m][l][i] += A[m][j] * B[j][i][l];
                }
            }
        }
    }
    end = clock();
    printf("c, k, b, a ---->>  CPU에서 행렬곱 실행시간 : % .8f second\n", (float)(end - start) / CLOCKS_PER_SEC);
    zeroMat(C, a, b, c);
    //k,a,b,c
    start = clock();
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < a; j++) {
            for (int l = 0; l < b; l++) {
                for (int m = 0; m < c; m++) {
                    C[j][l][m] += A[j][i] * B[i][m][l];
                }
            }
        }
    }
    end = clock();
    printf("k, a, b, c ---->>  CPU에서 행렬곱 실행시간 : % .8f second\n", (float)(end - start) / CLOCKS_PER_SEC);
    zeroMat(C, a, b, c);
    //k,a,c,b
    start = clock();
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < a; j++) {
            for (int l = 0; l < c; l++) {
                for (int m = 0; m < b; m++) {
                    C[j][m][l] += A[j][i] * B[i][l][m];
                }
            }
        }
    }
    end = clock();
    printf("k, a, c, b ---->>  CPU에서 행렬곱 실행시간 : % .8f second\n", (float)(end - start) / CLOCKS_PER_SEC);
    zeroMat(C, a, b, c);
    //k,b,a,c
    start = clock();
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < b; j++) {
            for (int l = 0; l < a; l++) {
                for (int m = 0; m < c; m++) {
                    C[l][j][m] += A[l][i] * B[i][m][j];
                }
            }
        }
    }
    end = clock();
    printf("k, b, a, c ---->>  CPU에서 행렬곱 실행시간 : % .8f second\n", (float)(end - start) / CLOCKS_PER_SEC);
    zeroMat(C, a, b, c);
    //k,b,c,a
    start = clock();
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < b; j++) {
            for (int l = 0; l < c; l++) {
                for (int m = 0; m < a; m++) {
                    C[m][j][l] += A[m][i] * B[i][l][j];
                }
            }
        }
    }
    end = clock();
    printf("k, b, c, a ---->>  CPU에서 행렬곱 실행시간 : % .8f second\n", (float)(end - start) / CLOCKS_PER_SEC);
    zeroMat(C, a, b, c);
    //k,c,a,b
    start = clock();
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < c; j++) {
            for (int l = 0; l < a; l++) {
                for (int m = 0; m < b; m++) {
                    C[l][m][j] += A[l][i] * B[i][j][m];
                }
            }
        }
    }
    end = clock();
    printf("k, c, a, b ---->>  CPU에서 행렬곱 실행시간 : % .8f second\n", (float)(end - start) / CLOCKS_PER_SEC);
    zeroMat(C, a, b, c);
    //k,c,b,a
    start = clock();
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < c; j++) {
            for (int l = 0; l < b; l++) {
                for (int m = 0; m < a; m++) {
                    C[m][l][j] += A[m][i] * B[i][j][l];
                }
            }
        }
    }
    end = clock();
    printf("k, c, b, a ---->>  CPU에서 행렬곱 실행시간 : % .8f second\n", (float)(end - start) / CLOCKS_PER_SEC);
    zeroMat(C, a, b, c);*/

    Matrix3D mA, mB, mC;
    mA.height = mA.stride_h = a; mA.width = mA.stride_w = k; mA.depth = 0;
    mB.height = mB.stride_h = c; mB.width = mB.stride_w = b; mB.depth = k;
    mC.height = mC.stride_h = b; mC.width = mC.width = c; mC.depth = a;
    mA.elements = (double*)malloc(sizeof(double) * a * k);
    mB.elements = (double*)malloc(sizeof(double) * k * c * b);
    mC.elements = (double*)malloc(sizeof(double) * a * b * c);
    
    for (int i = 0; i < a; i++) {
        for (int j = 0; j < k; j++) {
            mA.elements[i * k + j] = A[i][j];
        }
    }
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < c; j++) {
            for (int l = 0; l < b; l++) {
                mB.elements[c * b * i + b * j + l] = B[i][j][l];
            }
        }
    }

    matGPU2(mA, mB, mC);

    for (int i = 0; i < a; i++) {
        for (int j = 0; j < b; j++) {
            for (int l = 0; l < c; l++) {
                printf("%.3lf \n", C[i][j][l]);
            }
        }
    }
    printf("----------------------------\n");
    for (int i = 0; i < a; i++) {
        for (int j = 0; j < b; j++) {
            for (int l = 0; l < c; l++) {
                printf("%.3lf \n", mC.elements[b*c*i+c*j+l]);
            }
        }
    }
    

    for (int i = 0; i < a; i++) {
        for (int j = 0; j < b; j++) {
            for (int l = 0; l < c; l++) {
                if (mC.elements[b * c * i + c * j + l] != C[i][j][l]) printf("오류\n");
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


    free(mA.elements);
    free(mB.elements);
    free(mC.elements);


	return EXIT_SUCCESS;
}


void zeroMat(double*** M, const int a, const int b, const int c) {
    for (int i = 0; i < a; i++) {
        for (int j = 0; j < b; j++) {
            for (int k = 0; k < c; k++) {
                M[i][j][k] = 0;
            }
        }
    }
}



#define BLOCK_SIZE 16


__device__ double GetElement(const Matrix3D subA, int row, int col, int depth)
{
    return subA.elements[subA.stride_h * subA.stride_w * depth + subA.stride_w * row + col];
}


__device__ void SetElement(Matrix3D subA, int row, int col, int depth, double value)
{
    subA.elements[subA.stride_h * subA.stride_w * depth + subA.stride_w * row + col] = value;
}


__device__ Matrix3D GetSubMatrix(Matrix3D A, int block_row, int block_col,int block_depth)
{
    Matrix3D Asub;
    Asub.height = BLOCK_SIZE;
    Asub.width = BLOCK_SIZE;
    Asub.depth = BLOCK_SIZE;
    Asub.stride_h = A.stride_h;
    Asub.stride_w = A.stride_w;
    Asub.elements = &A.elements[A.stride_h * A.stride_w * BLOCK_SIZE * block_depth + A.stride_w * BLOCK_SIZE * block_row + BLOCK_SIZE * block_col];

    return Asub;
}


__global__ void MatMulKernel2(const Matrix3D A,const Matrix3D B, Matrix3D C) {
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int blockdepth = blockIdx.z;
    int row = threadIdx.y;
    int col = threadIdx.x;
    int depth = threadIdx.z;
    int x = BLOCK_SIZE * blockCol + col;
    int y = BLOCK_SIZE * blockRow + row;
    int z = BLOCK_SIZE * blockdepth + depth;

    Matrix3D Csub = GetSubMatrix(C, blockRow, blockCol,blockdepth);

    __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE][BLOCK_SIZE];

    double Cvalue = 0.0;

    for (int k = 0; k < (A.width + BLOCK_SIZE - 1) / BLOCK_SIZE; k++) {
        Matrix3D Asub = GetSubMatrix(A, blockRow, k, 0);
        Matrix3D Bsub = GetSubMatrix(B, k, blockCol, blockdepth);

        As[row][col] = 0.0;
        Bs[depth][row][col] = 0.0;

        if (k == ((A.width + BLOCK_SIZE - 1) / BLOCK_SIZE - 1)) {
            if (y < A.height && (col < A.width - k * BLOCK_SIZE)) As[row][col] = GetElement(Asub, row, col,0);
            if (x < B.width && y < B.height &&(depth < A.width - k * BLOCK_SIZE)) Bs[depth][row][col] = GetElement(Bsub, row, col, depth);
        }
        else {
            if (y < A.height) As[row][col] = GetElement(Asub, row, col,0);
            if (x < B.width && y < B.height) Bs[depth][row][col] = GetElement(Bsub, row, col, depth);
        }
        __syncthreads();

        for (int e = 0; e < BLOCK_SIZE; e++) {
            Cvalue += As[row][e] * Bs[depth][row][col];
        }

        __syncthreads();
    }

    if (x < C.width && y < C.height && z < C.depth) {
        SetElement(Csub, row, col, depth, Cvalue);
    }
}


void matGPU2(const Matrix3D A, const Matrix3D B, Matrix3D C) {
    Matrix3D d_A, d_B, d_C;
    d_A.width = d_A.stride_w = A.width; d_A.height = d_A.stride_h = A.height; d_A.depth = A.depth;
    d_B.width = d_B.stride_w = B.width; d_B.height = d_B.stride_h = B.height; d_B.depth = B.depth;
    d_C.width = d_C.stride_w = C.width; d_C.height = d_C.stride_h = C.height; d_C.depth = C.depth;


    // memory allocation
    cudaMalloc(&d_A.elements, sizeof(double) * d_A.width * d_A.height);
    cudaMalloc(&d_B.elements, sizeof(double) * d_B.width * d_B.height * d_B.depth);
    cudaMalloc(&d_C.elements, sizeof(double) * d_C.width * d_C.height * d_C.depth);
    cudaMemset(&d_C.elements, 0, sizeof(double) * d_C.width * d_C.height * d_C.depth);

    cudaMemcpy(&d_A, A.elements, sizeof(double) * d_A.width * d_A.height, cudaMemcpyHostToDevice);
    cudaMemcpy(&d_B, B.elements, sizeof(double) * d_B.width * d_B.height * d_B.depth, cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((d_C.width + dimBlock.x - 1) / dimBlock.x, (d_C.height + dimBlock.y - 1) / dimBlock.y, (d_C.depth + dimBlock.z - 1) / dimBlock.z);

    // call kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    MatMulKernel2 << <dimGrid, dimBlock >> > (d_A, d_B, d_C);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU에서 행렬곱 실행시간 : % .8f second\n", milliseconds / 1000.);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(C.elements, d_C.elements, sizeof(double) * d_C.width * d_C.height * d_C.depth, cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);

    return;
}