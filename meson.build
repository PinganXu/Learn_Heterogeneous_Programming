project('Learn_CUDA_Programming', ['cuda'])

'''
    I have a RTX 2060 Super GPU, its architecture name is Turing with 
    compute compability compute_75. I want the generated binary code 
    runs on this GPU, so I set the args as '-code=75'.
    If you have other GPUs, you can change the args as below.
'''
cuda_compile_args = ['-arch=compute_75', '-code=sm_75']
cuda_compile_dep = dependency('cuda', version: '>=12.0', modules: ['cudart'])

# src zone
executable('case1_hello_world', 'src/case1_helloworld/case1_helloworld.cu')
executable('topic1_mma_16_16_16', 'src/topic1_tensor_core/case1_mma_16_16_16.cu', cuda_args: cuda_compile_args, dependencies: cuda_compile_dep)
executable('topic2_case1_constant_mem_test', 'src/topic2_memory/case2_1_constant_memory.cu', cuda_args: cuda_compile_args, dependencies: cuda_compile_dep)

# ptx_src zone
executable('ptx_vector_add', 'ptx_src/vector_add/vector_add_host.cu', cuda_args: cuda_compile_args, dependencies: cuda_compile_dep)
