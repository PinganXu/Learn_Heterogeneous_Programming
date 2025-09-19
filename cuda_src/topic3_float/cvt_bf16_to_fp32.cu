#include <iostream>
#include <assert.h>
#include <cuda_bf16.h>

using namespace std;

__global__ void test_bf16_to_fp32(void) {
    unsigned thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint16_t idx = 0; idx < (1 << 2); idx++) {
        uint16_t src = (thread_idx << 2) + idx;
        float golden_float = __bfloat162float(*((__nv_bfloat16*)&src));
        uint32_t golden = *((uint32_t*)&golden_float);
        uint32_t res = 0;
        res = src << 16;
        assert(res == golden);
    }
}

int main(void) { 
	test_bf16_to_fp32<<<64, 256>>>();
	cudaDeviceSynchronize();
    return 0; 
}