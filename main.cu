#include "matMul.cuh"
#include<stdio.h>
#include<stdlib.h>
#include<time.h>


void compare(const float*, const Matrix, const int);


int main() {
    // m == a.height, n == b.width, k == a.width
    int m, n, k;
    srand((unsigned int)time(NULL));
    m = 110;//rand() + 16;
    n = 110;//rand() + 16;
    k = 110;//rand() + 16;

    int size_A = m * k * sizeof(float);
    int size_B = n * k * sizeof(float);
    int size_C = m * n * sizeof(float);

    // memory allocation
    float* c_A, * c_B, * c_C;
    c_A = (float*)malloc(size_A);
    c_B = (float*)malloc(size_B);
    c_C = (float*)malloc(size_C);

    Matrix d_A, d_B, d_C;
    d_A.height = m; d_A.width = d_A.stride = k;
    d_B.height = k; d_B.width = d_B.stride = n;
    d_C.height = m; d_C.width = d_C.stride = n;
    d_A.elements = (float*)malloc(size_A);
    d_B.elements = (float*)malloc(size_B);
    d_C.elements = (float*)malloc(size_C);


    // fill matrix A, B
    setMatrix(c_A, m * k);
    setMatrix(c_B, k * n);
    for (int i = 0; i < m * k; i++) {
        d_A.elements[i] = c_A[i];
    }
    for (int i = 0; i < n * k; i++) {
        d_B.elements[i] = c_B[i];
    }


    // matrix multiplication
    clock_t start1, end1;
    
    start1 = clock();
    matCPU(c_A, c_B, c_C, m, n, k);
    end1 = clock();
    printf("CPU에서 행렬곱 실행시간 : % .3f\n", (float)(end1 - start1) / CLOCKS_PER_SEC);
    
    //printCPU(c_C, m, n);
    printf("\n");


    matGPU(d_A, d_B, d_C);
  
    //printGPU(d_C);
    printf("\n");


    // compare matrix
    compare(c_C, d_C, m * n);


    // free memory
    free(c_A);
    free(c_B);
    free(c_C);
    free(d_A.elements);
    free(d_B.elements);
    free(d_C.elements);

    return 0;
}

// compare matrix
void compare(const float* c_m, const Matrix d_m, const int size) {
    for (int i = 0; i < size; i++) {
        if (c_m[i] != d_m.elements[i]) {
            printf("일치하지 않는 부분 : c_C[%d] = %.3f, d_C[%d] = %.3f\n", i, c_m[i], i, d_m.elements[i]);   
        }
    }

    printf("두 행렬이 일치합니다.\n");
    return;
}