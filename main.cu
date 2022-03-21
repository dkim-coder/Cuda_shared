#include "matMul.cuh"
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#define _CTR_SEUCRE_NO_WARNINGS

void compare(const Matrix, const Matrix);


int main() {
    // m == a.height, n == b.width, k == a.width
    int m, n, k;
    
    printf("input a.height : ");
    scanf("%d", &m);
    printf("input b.width : ");
    scanf("%d", &n);
    printf("input a.width : ");
    scanf("%d", &k);

    size_t size_A = m * k * sizeof(double);
    size_t size_B = n * k * sizeof(double);
    size_t size_C = m * n * sizeof(double);

    // memory allocation
    Matrix A, B, C1, C2, C3;
    A.width = A.stride = k; A.height = m;
    B.width = B.stride = n; B.height = k;
    C1.width = C1.stride = n; C1.height = m;
    C2.width = C2.stride = n; C2.height = m;
    C3.width = C3.stride = n; C3.height = m;
    A.elements = (double*)malloc(size_A);
    B.elements = (double*)malloc(size_B);
    C1.elements = (double*)malloc(size_C);
    C2.elements = (double*)malloc(size_C);
    C3.elements = (double*)malloc(size_C);

    // fill matrix A, B
    setMatrix(A);
    setMatrix(B);

    // matrix multiplication on CPU
    clock_t start1, end1;
    start1 = clock();
    matCPU(A, B, C1);
    end1 = clock();
    printf("CPU에서 행렬곱 실행시간 : % .8f second\n", (float)(end1 - start1) / CLOCKS_PER_SEC);
    printf("\n");

    // matrix multiplication on GPU
    matGPU(A, B, C2);    
    printf("\n");


    // matrix multiplication using cublas
    cublasMat(A, B, C3);
    printf("\n");


    // compare matrix
    compare(C1, C2);
    compare(C1, C3);
    

    //printMatrix(C1);
    //printMatrix(C2);
    //printMatrix(C3);
    

    // free memory
    free(A.elements);
    free(B.elements);
    free(C1.elements);
    free(C2.elements);
    free(C3.elements);

    return 0;
}


// compare matrix
void compare(const Matrix A, const Matrix B) {
    bool r = true;
    
    for (int i = 0; i < (A.height * A.width); i++) {
        double diff = A.elements[i] - B.elements[i];
        if (diff < 0) { diff *= -1;  }

        // 내가 만든 커널이 정밀도? 는 더 높다 0 개수 차이
        if (diff > 0.00000000001) {
            printf("일치하지 않는 부분 : CPU[%d] = %.15lf, GPU[%d] = %.15lf\n", i, A.elements[i], i, B.elements[i]);
            r = false;
        }
    }

    if (r == true) {
        printf("두 행렬이 일치합니다.\n");
    }
    
    return;
}