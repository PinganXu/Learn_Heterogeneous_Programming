
import typing
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr: int,
               y_ptr: int,
               output_ptr: int,
               n_elements: int,
               BLOCK_SIZE: tl.constexpr, ) -> None:
    blockIdx = tl.program_id(axis=0)  # program_id is like the block idx in a grid
    '''
        There has no threadIdx in triton, triton use vector operation instead of 
        thread index, for example:
            offsets = blockIdx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    '''
    offsets = blockIdx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = tl.add(x, y)
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # step 1: preallocate the output memory in device, actually, torch has the feature
    output: torch.Tensor = torch.empty_like(x)  # because output has same size as input
    assert x.is_cuda and y.is_cuda and output.is_cuda  # keep all data calculate in gpu

    n_elements: int = output.numel()

    # the lambda function returns a tuple, indicate grid size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    
    # lambda function grid will get parameter 'BLOCK_SIZE'
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output



def main() -> None:
    # the vector has 1M elements, if dtype == float, cost 4MB memory space
    vector_len: int = 1024 * 1024
    vec_A: torch.Tensor = torch.rand(vector_len, dtype=torch.float32, device='cuda')
    vec_B: torch.Tensor = torch.rand(vector_len, dtype=torch.float32, device='cuda')
    print('vec_A = ', vec_A)
    print('vec_B = ', vec_B)

    '''
        Plan 1: the easy way to do vector add is to use '+' as default operation in torch.
        The whole execution is list as below:
            1.  Tensor operator '+' acts like a function call, it will call the cuda backend,
                then treat vec_A and vec_B as kernel arg, and use xdma to move the data from 
                host memory to device memory.
            2.  Also, the device memory space of vec_C will be allocated by nvidia runtime api,
                and then launch the kernel function 'vector_add', when kernel has been finished,
                xdma do data movement from device memory to host memory.
    ''' 
    vec_C: torch.Tensor = vec_A + vec_B
    print('gloden: torch res =', vec_C)

    '''
        Plan 2: Sometimes, we want a high performance to get the result, a directly method is use
                python to write kernel instead of function call of torch.
                That is why we use triton!
    '''
    vec_C_triton: torch.Tensor = add(vec_A, vec_B)
    print('triton res =', vec_C_triton)


if __name__ == '__main__':
    main()