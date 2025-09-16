#include <iostream>
#include <cuda_fp4.h>

using namespace std;

__global__ void test_fp32_to_fp4(void) {
    unsigned thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t idx = 0; idx < (1 << 18); idx++) {
        uint32_t src = (thread_idx << 18) + idx;
        __nv_fp4_storage_t golden = __nv_cvt_float_to_fp4(*((float*)&src), __NV_E2M1, cudaRoundNearest);
        char res = 0x0U;
        uint32_t sign = src >> 31;
        uint32_t absx = src & 0x7fffffffU;
        if (absx <= 0x3e800000U) {
            // <= 0.25, res = {s, 0b000}
            res = sign << 3;
        }
        else if (absx > 0x3e800000U && absx < 0x3f400000U) {
            // > 0.25 && < 0.75, res = {s, 0b001}
            res = (sign << 3) | 0x1U;
        }
        else if (absx >= 0x3f400000U && absx <= 0x3fa00000U) {
            // >= 0.75 && <= 1.25, res = {s, 0b010}
            res = (sign << 3) | 0x2U;
        }
        else if (absx > 0x3fa00000U && absx < 0x3fe00000U) {
            // > 1.25 && < 1.75, res = {s, 0b011}
            res = (sign << 3) | 0x3U;
        }
        else if (absx >= 0x3fe00000U && absx <= 0x40200000U) {
            // >= 1.75 && <= 2.5, res = {s, 0b100}
            res = (sign << 3) | 0x4U;
        }
        else if (absx > 0x40200000U && absx < 0x40600000U) {
            // > 2.5 && < 3.5, res = {s, 0b101}
            res = (sign << 3) | 0x5U;
        }
        else if (absx >= 0x40600000U && absx <= 0x40a00000U) {
            // >= 3.5 && <= 5, res = {s, 0b110}
            res = (sign << 3) | 0x6U;
        }
        else { // absx > 0x40a00000U
            uint32_t exp = (src >> 23) & 0xff;
            uint32_t man = src & 0x7fffff;
            sign = (exp == 0xff && man != 0) ? 0x0U : sign;
            res = (sign << 3) | 0x7U;
        }
        assert(res == golden);
    }
}

int main(void) { 
	test_fp32_to_fp4<<<64, 256>>>();
	cudaDeviceSynchronize();
    return 0; 
}