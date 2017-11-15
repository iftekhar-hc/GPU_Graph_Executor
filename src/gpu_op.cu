#include "./c_runtime_api.h"
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CHECK_GPU_ERR(ans) { gpu_assert((ans), __FILE__, __LINE__); }

inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPU_assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* TODO: Your code here */
/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void matrix_softmax_cross_entropy_kernel(int nrow, int ncol,
                                                    const float *input_a,
                                                    const float *input_b,
                                                    float *output) {
  // Dynamic shared memory, size provided at kernel launch.
  extern __shared__ float loss_per_row[];
  // Two dimensional thread blocks.
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
          threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input_a += y * ncol;
  input_b += y * ncol;
  float maxval = *input_a;
  // Find max for a row.
  for (int x = 1; x < ncol; ++x) {
    maxval = max(maxval, input_a[x]);
  }
  // Deduct by max for a row, and raise to exp.
  float sum = 0;
  for (int x = 0; x < ncol; ++x) {
    sum += exp(input_a[x] - maxval);
  }
  // Compute per-row loss.
  float loss = 0;
  for (int x = 0; x < ncol; ++x) {
    loss -= input_b[x] * log(exp(input_a[x] - maxval) / sum);
  }
  loss_per_row[y] = loss;
  __syncthreads();
  // Compute reduce_mean across rows.
  float mean_loss = 0;
  // Use a single thread to reduce mean across rows.
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    for (int i = 0; i < nrow; ++i) {
      mean_loss += loss_per_row[i];
    }
    mean_loss /= nrow;
    output[0] = mean_loss;
  }
}

__global__ void matrix_elementwise_add_kernel(int nrow, int ncol,
                                              const float *matA,
                                              const float *matB,
                                              float *output) {
#if 1
  int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int col_idx = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = row_idx * ncol;
  // int idx = col_idx + row_idx * ncol;
  // printf("row_idx = %d, col_idx = %d\n", row_idx, col_idx);
  // printf("tid = %d ", idx);
  if (row_idx < nrow && col_idx < ncol) {
    for (size_t i = col_idx; i < ncol; ++i) {
      output[idx + i] = matA[idx + i] + matB[idx + i];
    }
  }
#endif
}

__global__ void matrix_elementwise_add_by_const_kernel(int nrow, int ncol,
                                                       const float *input,
                                                       const float val,
                                                       float *output) {
#if 0
  // int block_idx = blockIdx.x * gridDim.x + threadIdx.x;
  int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int col_idx = blockIdx.y * blockDim.y + threadIdx.y;
  printf("row_idx = %d, col_idx = %d\n", row_idx, col_idx);
  int idx = row_idx * ncol;
  if (row_idx < nrow && col_idx < ncol) {
    for (size_t i = 0; i < ncol; ++i)
      output[idx + i] = input[idx + i] + val;
  }
#endif
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
  printf("tid_x = %d, tid_y = %d\n", tid_x, tid_y);
  int idx = tid_y * ncol + tid_x;
  if (tid_x < ncol && tid_y < nrow) {
     output[idx] = input[idx] + val;
  }
}

__global__ void matrix_elementwise_multiply_kernel(int nrow, int ncol,
                                                   const float *matA,
                                                   const float *matB,
                                                   float *output) {
  int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int col_idx = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = row_idx * ncol;
  if (row_idx < nrow && col_idx < ncol) {
    for (size_t i = 0; i < ncol; ++i)
      output[idx + i] = matA[idx + i] * matB[idx + i];
  }
}

__global__ void matrix_elementwise_multiply_by_const_kernel(int nrow, int ncol,
                                                            const float *input,
                                                            const float val,
                                                            float *output) {
  int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int col_idx = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = row_idx * ncol;
  if (row_idx < nrow && col_idx < ncol) {
    for (size_t i = 0; i < ncol; ++i)
      output[idx + i] = input[idx + i] * val;
  }
}

__global__ void array_set_kernel(int nrow, int ncol,
                                float *arr,
                                const float val) {
  int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int col_idx = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = row_idx * ncol;
  if (row_idx < nrow && col_idx < ncol) {
    for (size_t i = 0; i < ncol; ++i)
      arr[idx + i] = val;
  }
}

int DLGpuArraySet(DLArrayHandle arr, float value) { 
  /* TODO: Your code here */
  assert(arr->ndim == 2);
  int nrow = arr->shape[0];
  // Maximum x- or y-dimension of a block = 1024
  // But we need 'nrow' shared memory, and max shared memory is 48KB.
  // Conservatively allow max 16KB shared memory.
  assert(nrow <= 1024 * 4);
  int ncol = arr->shape[1];
  float *input_data = (float *)arr->data;
  dim3 threads;
  if (nrow * ncol <= 1024) {
    threads.x = nrow;
    threads.y = ncol;
  } else {
    threads.x = nrow;
    threads.y = (nrow + 1023) / 1024;
  }
  // 1 block, each block with 'threads' number of threads with 'nrow' shared
  // memory size
  array_set_kernel<<<1, threads>>>(nrow, 
                                   ncol, 
                                   input_data, 
                                   value);
  return 0;
}

int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  return 0;
}

int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  return 0;
}

int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                              const DLArrayHandle matB, DLArrayHandle output) {
  /* TODO: Your code here */
  assert(matA->ndim == 2);
  assert(matB->ndim == 2);
  assert(output->ndim == 2);
  assert(matA->shape[0] == matB->shape[0] &&
         matA->shape[1] == matB->shape[1]); 
  int nrow = matA->shape[0];
  // Maximum x- or y-dimension of a block = 1024
  // But we need 'nrow' shared memory, and max shared memory is 48KB.
  // Conservatively allow max 16KB shared memory.
  assert(nrow <= 1024 * 4);
  int ncol = matA->shape[1];
  const float *matA_data = (const float *)matA->data;
  const float *matB_data = (const float *)matB->data;
  float *output_data = (float *)output->data;
  dim3 threads;
  if (nrow * ncol <= 1024) {
    threads.x = nrow;
    threads.y = ncol;
  } else {
    threads.x = nrow;
    threads.y = (nrow + 1023) / 1024;
    // threads.y = ncol;
  }
  // 1 block, each block with 'threads' number of threads with 'nrow' shared
  // memory size
  matrix_elementwise_add_kernel<<<1, threads>>>(
      nrow, ncol, matA_data, matB_data, output_data);
  CHECK_GPU_ERR( cudaPeekAtLastError() );
  CHECK_GPU_ERR( cudaDeviceSynchronize() );
  return 0;
}

int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                     DLArrayHandle output) {
  /* TODO: Your code here */
  assert(input->ndim == 2);
  assert(output->ndim == 2);
  int nrow = input->shape[0];
  // Maximum x- or y-dimension of a block = 1024
  // But we need 'nrow' shared memory, and max shared memory is 48KB.
  // Conservatively allow max 16KB shared memory.
  assert(nrow <= 1024 * 4);
  int ncol = input->shape[1];
  const float *input_data = (const float *)input->data;
  float *output_data = (float *)output->data;
  dim3 threads;
  dim3 blocks;
  if (nrow * ncol <= 1024) {
    threads.x = nrow;
    threads.y = ncol;
  } else {
    blocks.x = (ncol + 1024 - 1) / 1024;
    blocks.y = (nrow + 1024 - 1) / 1024;
    // blocks.x = ceil(nrow/1024.0);
    // blocks.y = ceil(ncol/1024.0);
    threads.x = min(nrow, 1024);
    // threads.y = (nrow + 1023) / 1024;
    threads.y = 1;
  }
  // 1 block, each block with 'threads' number of threads with 'nrow' shared
  // memory size
  matrix_elementwise_add_by_const_kernel<<<blocks, threads>>>(
      nrow, ncol, input_data, val, output_data);
  CHECK_GPU_ERR( cudaPeekAtLastError() );
  CHECK_GPU_ERR( cudaDeviceSynchronize() );
  return 0;
}

int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA,
                                   const DLArrayHandle matB,
                                   DLArrayHandle output) {
  /* TODO: Your code here */
  assert(matA->ndim == 2);
  assert(matB->ndim == 2);
  assert(output->ndim == 2);
  assert(matA->shape[0] == matB->shape[0] &&
         matA->shape[1] == matB->shape[1]); 
  int nrow = matA->shape[0];
  // Maximum x- or y-dimension of a block = 1024
  // But we need 'nrow' shared memory, and max shared memory is 48KB.
  // Conservatively allow max 16KB shared memory.
  assert(nrow <= 1024 * 4);
  int ncol = matA->shape[1];
  const float *matA_data = (const float *)matA->data;
  const float *matB_data = (const float *)matB->data;
  float *output_data = (float *)output->data;
  dim3 threads;
  if (nrow * ncol <= 1024) {
    threads.x = nrow;
    threads.y = ncol;
  } else {
    threads.x = nrow;
    threads.y = (nrow + 1023) / 1024;
  }
  // 1 block, each block with 'threads' number of threads with 'nrow' shared
  // memory size
  matrix_elementwise_multiply_kernel<<<1, threads, nrow * sizeof(float)>>>(
      nrow, ncol, matA_data, matB_data, output_data);
  return 0;
}

int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                               DLArrayHandle output) {
  /* TODO: Your code here */
  assert(input->ndim == 2);
  assert(output->ndim == 2);
  int nrow = input->shape[0];
  // Maximum x- or y-dimension of a block = 1024
  // But we need 'nrow' shared memory, and max shared memory is 48KB.
  // Conservatively allow max 16KB shared memory.
  assert(nrow <= 1024 * 4);
  int ncol = input->shape[1];
  const float *input_data = (const float *)input->data;
  float *output_data = (float *)output->data;
  dim3 threads;
  if (nrow * ncol <= 1024) {
    threads.x = nrow;
    threads.y = ncol;
  } else {
    threads.x = nrow;
    threads.y = (nrow + 1023) / 1024;
  }
  // 1 block, each block with 'threads' number of threads with 'nrow' shared
  // memory size
  matrix_elementwise_multiply_by_const_kernel<<<1, threads, nrow * sizeof(float)>>>(
      nrow, ncol, input_data, val, output_data);
  return 0;
}

int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                        const DLArrayHandle matB, bool transposeB,
                        DLArrayHandle matC) {
  /* TODO: Your code here */
  // Hint: use cublas
  // cublas assume matrix is column major
  return 0;
}

int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  return 0;
}

int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                      DLArrayHandle output) {
  /* TODO: Your code here */
  return 0;
}

int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  return 0;
}

int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
                             const DLArrayHandle input_b,
                             DLArrayHandle output) {
  assert(input_a->ndim == 2);
  assert(input_b->ndim == 2);
  assert(output->ndim == 1);
  assert(input_a->shape[0] == input_b->shape[0] &&
         input_a->shape[1] == input_b->shape[1]);
  int nrow = input_a->shape[0];
  // Maximum x- or y-dimension of a block = 1024
  // But we need 'nrow' shared memory, and max shared memory is 48KB.
  // Conservatively allow max 16KB shared memory.
  assert(nrow <= 1024 * 4);
  int ncol = input_a->shape[1];
  const float *input_data_a = (const float *)input_a->data;
  const float *input_data_b = (const float *)input_b->data;
  float *output_data = (float *)output->data;
  dim3 threads;
  if (nrow <= 1024) {
    threads.x = nrow;
  } else {
    threads.x = 1024;
    threads.y = (nrow + 1023) / 1024;
  }
  // 1 block, each block with 'threads' number of threads with 'nrow' shared
  // memory size
  matrix_softmax_cross_entropy_kernel<<<1, threads, nrow * sizeof(float)>>>(
      nrow, ncol, input_data_a, input_data_b, output_data);
  return 0;
}
