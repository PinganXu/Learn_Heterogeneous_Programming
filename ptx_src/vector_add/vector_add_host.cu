#include <iostream>
#include <cuda_runtime.h>

__global__ void vector_add(float* src_A, float* src_B, float* src_C, unsigned N) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    // use ptx to do vector add
    asm volatile (
        "{\n\t"
        "    .reg .f32 a, b;                    \n\t"
        "    .reg .u64 addr_a, addr_b, addr_c;  \n\t"
        "    mul.wide.u32 addr_a, %3, 4;        \n\t"
        "    add.u64 addr_b, addr_a, %1;        \n\t"
        "    add.u64 addr_c, addr_a, %2;        \n\t"
        "    add.u64 addr_a, addr_a, %0;        \n\t"
        "    ld.global.f32 a, [addr_a];         \n\t"
        "    ld.global.f32 b, [addr_b];         \n\t"
        "    add.f32 a, a, b;                   \n\t"
        "    st.global.f32 [addr_c], a;         \n\t"
        "}\n\t"
        :                                               // no output
        : "l"(src_A), "l"(src_B), "l"(src_C), "r"(idx)  // %0, %1, %2, %3
        : "memory"
    );
}


int main() {
    int size = 1024;
    float *h_A = new float[size], *h_B = new float[size], *h_C = new float[size];
    float *d_A, *d_B, *d_C;

    for (int i = 0; i < size; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    // 分配设备内存
    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, size * sizeof(float));
    cudaMalloc(&d_C, size * sizeof(float));

    // 复制数据到设备
    cudaMemcpy(d_A, h_A, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size * sizeof(float), cudaMemcpyHostToDevice);

    // 调用PTX内核
    vector_add<<<8, 128>>>(d_A, d_B, d_C, size); // 4个块，256线程

    // 复制结果回主机
    cudaMemcpy(h_C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);

    // 验证结果（示例）
    printf("C[0]=%.1f, C[511]=%.1f\n", h_C[0], h_C[511]); // 应为0+0=0, 511+1022=1533

    // 释放资源
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;
    return 0;
}

