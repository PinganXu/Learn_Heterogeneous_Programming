#include <iostream>
#include <cuda_fp4.h>

using namespace std;

__global__ void execute_on_gpu(void) {
    // unsigned cal_idx = blockIdx.x * blockDim.x + threadIdx.x;
    // unsigned x = 0U | (0x7dU << 23) | (0x600000U - 1U);
    // unsigned x = 0U | (0x7dU << 23);
    for (unsigned x = 0x7dU << 23; x <= 0xffffffff; x++) {
        __nv_fp4_storage_t res = __nv_cvt_float_to_fp4(*((float*)&x), __NV_E2M1, cudaRoundNearest);
        if (res != 0) {
            printf("%x, %f \n", x, *((float*)&x));
            printf("%d\n", (unsigned)res);
            assert(false);
        }
    }
}

int main(void) { 
	execute_on_gpu<<<1, 1>>>();
	cudaDeviceSynchronize();
    return 0; 
}