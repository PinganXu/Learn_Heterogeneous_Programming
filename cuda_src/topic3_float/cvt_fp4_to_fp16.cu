#include <iostream>
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>

using namespace std;

__global__ void test_fp4_to_fp16(void) {
    for (uint8_t src = 0; src < (1 << 4); src++) { 
        __half_raw res = __nv_cvt_fp4_to_halfraw(*((__nv_fp4_storage_t*)&src), __NV_E2M1);
        printf("res = 0x%x, src = 0x%x, abs = 0x%x\n", *((uint16_t*)&res), (uint16_t)src, *((uint16_t*)&res) & 0x7fffU);
    }
}

int main(void) { 
	test_fp4_to_fp16<<<1, 1>>>();
	cudaDeviceSynchronize();
    return 0; 
}