#include <iostream>

using namespace std;

__global__ void execute_on_gpu(void) {
    if (threadIdx.x % 32 == 0) {
        printf("hello world from gpu block_%d, thread_%d\n", 
                blockIdx.x, threadIdx.x);
    }
}

int main(void) { 
	std::cout << "hello world from host" << std::endl;
	execute_on_gpu<<<2, 512>>>();
	cudaDeviceSynchronize();
    return 0; 
}
