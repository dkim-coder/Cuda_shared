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

    int size_A = m * k * sizeof(float);
    int size_B = n * k * sizeof(float);
    int size_C = m * n * sizeof(float);

    // memory allocation
    Matrix A, B, C1, C2;
    A.width = A.stride = k; A.height = m;
    B.width = B.stride = n; B.height = k;
    C1.width = C1.stride = n; C1.height = m;
    C2.width = C2.stride = n; C2.height = m;
    A.elements = (float*)malloc(size_A);
    B.elements = (float*)malloc(size_B);
    C1.elements = (float*)malloc(size_C);
    C2.elements = (float*)malloc(size_C);

    // fill matrix A, B
    setMatrix(A);
    setMatrix(B);


    // matrix multiplication on CPU
    clock_t start1, end1;
    start1 = clock();
    matCPU(A, B, C1);
    end1 = clock();
    printf("CPU���� ��İ� ����ð� : % .3f\n", (float)(end1 - start1) / CLOCKS_PER_SEC);
    printf("\n");

    // matrix multiplication on GPU
    matGPU(A, B, C2);
    printf("\n");


    // compare matrix
    compare(C1, C2);


    // free memory
    free(A.elements);
    free(B.elements);
    free(C1.elements);

    return 0;
}

// compare matrix
void compare(const Matrix A, const Matrix B) {
    for (int i = 0; i < (A.height * A.width); i++) {
        if ((A.elements[i] - B.elements[i]) > 0.00000001) {
            printf("��ġ���� �ʴ� �κ� : C1[%d] = %.8f, C2[%d] = %.8f\n", i, A.elements[i], i, B.elements[i]);   
        }
    }

    printf("�� ����� ��ġ�մϴ�.\n");
    return;
}