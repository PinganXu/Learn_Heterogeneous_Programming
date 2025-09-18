#include <iostream>
#include <assert.h>
#include <cuda_fp16.h>

using namespace std;

__global__ void test_fp16_to_fp32(void) {
    unsigned thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint16_t idx = 0; idx < (1 << 2); idx++) {
        uint16_t src = (thread_idx << 2) + idx;
        float golden_float = __half2float(*((__half*)&src));
        uint32_t golden = *((uint32_t*)&golden_float);
        uint32_t res = 0;
        uint16_t sign = src >> 15;
        uint16_t exp = (src >> 10) & 0x1fU;
        uint16_t man = src & 0x3ffU;

        if (exp == 0 && man == 0) {
            res |= sign << 31;
        }
        else if (exp == 0 &&  man != 0) {
            // subnormal
            unsigned man_ = man;
            unsigned nlz = 0;  // number of leading zeros  
            if ((man_ & 0xff00U) == 0) { nlz += 8; man_ <<= 8; }
            if ((man_ & 0xf000U) == 0) { nlz += 4; man_ <<= 4; }
            if ((man_ & 0xc000U) == 0) { nlz += 2; man_ <<= 2; }
            if ((man_ & 0x8000U) == 0) { nlz += 1; man_ <<= 1; }
            nlz -= 6;
            res |= (0x70U - nlz) << 23;  // 0x7fU - 0xfU -nlz
            res |= (man << (nlz + 14)) & 0x7fffffU;
            res |= sign << 31;
        }
        else if (exp == 0x1fU && man != 0) {
            // NAN
            res = 0x7fffffffU;  // qNAN
        }
        else if (exp == 0x1fU && man == 0) {
            // Inf
            res = 0x7f800000U;
            res |= sign << 31;
        }
        else {
            res |= (exp + 0x70U) << 23;
            res |= man << 13;
            res |= sign << 31;
        }
        assert(res == golden);
    }
}

int main(void) { 
	test_fp16_to_fp32<<<64, 256>>>();
	cudaDeviceSynchronize();
    return 0; 
}