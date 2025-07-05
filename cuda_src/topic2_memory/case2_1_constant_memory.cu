#include <iostream>
#include <cuda_runtime.h>

__constant__ unsigned c_mem[32];

__global__ void constant_memory_test(unsigned* result) {
    unsigned block_idx = blockIdx.x;
    unsigned thread_idx = threadIdx.x;

    result[block_idx * blockDim.x + thread_idx] = block_idx * blockDim.x + c_mem[thread_idx % 32];
}

int main() {
    unsigned Grid_size = 8;
    unsigned Block_size = 32;
    unsigned *h_mem = new unsigned[32];
    unsigned *h_res = new unsigned[Grid_size * Block_size];

    for (unsigned i = 0; i < 32; i++) {
        h_mem[i] = i + (1 << 10);
    }
    cudaMemcpyToSymbol(c_mem, h_mem, sizeof(unsigned) * 32);

    unsigned *d_res;
    cudaMalloc(&d_res, sizeof(unsigned) * Grid_size * Block_size);

    constant_memory_test<<<Grid_size, Block_size>>>(d_res);

    cudaMemcpy(h_res, d_res, sizeof(unsigned) * Grid_size * Block_size, cudaMemcpyKind::cudaMemcpyDeviceToHost);

    cudaFree(d_res);

    for (unsigned i = 0; i < Grid_size * Block_size; i++) {
        std::cout << h_res[i] << std::endl;
    }

    delete h_mem;
    delete h_res;

    return 0;
}
