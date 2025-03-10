project('Learn_CUDA_Programming', ['cpp', 'cuda'])

cuda_dep = dependency('cuda', version: '>=12.0', modules: ['cublas', 'cudart'])

executable('case1_hello_world', 'src/case1_helloworld/case1_helloworld.cu')
executable('topic1_mma_16_16_16', 'src/topic1_tensor_core/case1_mma_16_16_16.cu', dependencies: cuda_dep)
