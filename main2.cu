#include "matMul.cuh"
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define _CTR_SEUCRE_NO_WARNINGS

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

    int A[2][3][4] = {
        {
            {1, 2, 3, 4},
        {5,6,7,8}
        }
    {
        {1,2,3,4}
    }
    





	return EXIT_SUCCESS;
}