#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include <mma.h>

using namespace std;
using namespace nvcuda;

__global__ void mma_16_16_16(half* a, half* b, float* c) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // Initialize the output to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Load the inputs
    wmma::load_matrix_sync(a_frag, a, 16);
    wmma::load_matrix_sync(b_frag, b, 16);

    // Perform the matrix multiplication
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store the output
    wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
}

void gen_random_half(half* data, unsigned N) {
    std::random_device rd;
    std::mt19937 gen(rd());
    // generate random float number by normal distribution with mean=0.0f and var=1.0f
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (unsigned i = 0; i < N; i++) {
        data[i] = __float2half(dist(gen));
    }
}

void run_mma_16_16_16() {
    half *h_a, *h_b, *d_a, *d_b;
    float *h_c, *d_c;
    
    h_a = new half[16 * 16];
    h_b = new half[16 * 16];
    h_c = new float[16 * 16];

    gen_random_half(h_a, 16 * 16);
    gen_random_half(h_b, 16 * 16);
    
    cudaMalloc(&d_a, sizeof(half) * 16 * 16);
    cudaMalloc(&d_b, sizeof(half) * 16 * 16);
    cudaMalloc(&d_c, sizeof(float) * 16 * 16);

    cudaMemcpy(d_a, h_a, sizeof(half) * 16 * 16, cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(half) * 16 * 16, cudaMemcpyKind::cudaMemcpyHostToDevice);

    mma_16_16_16<<<1, 32>>>(h_a, h_b, h_c);

    cudaMemcpy(h_c, d_c, sizeof(float) * 16 * 16, cudaMemcpyKind::cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(h_a);
    cudaFree(h_b);
    cudaFree(h_c);

    std::cout << "print result as below:" << std::endl;
    for (unsigned i = 0; i < 16 * 16; i++) {
        std::cout << h_c[i] << "\t";
        if (i % 16 == 0)
            std::cout << std::endl;
    }

}

int main() {
    run_mma_16_16_16();
    return 0;
}
