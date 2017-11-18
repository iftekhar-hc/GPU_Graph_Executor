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

__global__ void softmax_kernel(int nrow,
                               int ncol,
                               const float *input,
                               float *output) {
#if 0
   size_t i = blockIdx.y;
   size_t j = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < nrow && j < ncol) {
     size_t idx = i * ncol + j;
     output[idx] = max(input[idx], 0.0);
   }
#endif
#if 0
   for (int i = 0; i < nrow; ++i)
     for (int j = 0; j < ncol; ++j)
       output[i * ncol + j] = max(input[i * ncol + j], 0.0);
#endif
   for (int i = 0; i < nrow; ++i) {
     float max_val = 1e-9;
     float sum = 0.0;
     for (int j = 0; j < ncol; ++j) {
       float cur_val = input[i * ncol + j]; 
       max_val = max(cur_val, max_val);
       sum += cur_val;
     }
     for (int j = 0; j < ncol; ++j) {
       size_t idx = i * ncol + j;
       float norm_val = input[idx] - max_val;
       output[idx] = exp(norm_val) / sum;
     }
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
#if 0
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
  // printf("tid_x = %d, tid_y = %d\n", tid_x, tid_y);
  printf("%d %d %d %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
  int idx = tid_y * ncol + tid_x;
  if (tid_x < ncol && tid_y < nrow) {
     output[idx] = input[idx] + val;
  }
#endif
  unsigned tid_x = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned tid_y = blockIdx.y;
  // printf("%d %d %d %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
  unsigned idx = tid_y * ncol + tid_x;
  if (tid_x < ncol && tid_y < nrow) {
     output[idx] = input[idx] + val;
  }
}

__global__ void matrix__multiply_kernel(const float *matA,
                                        const float *matB,
                                        float *matC,
                                        int nrow,
                                        int ncol,
                                        int width) {
  // int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  // int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
#if 0
  int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
  int tid_y = blockIdx.y;
  printf("%d %d %d %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
  float accum_val = 0.0f;
  if (tid_x < width && tid_y < width) {
    for (int k = 0; k < width; ++k) {
      accum_val += matA[tid_y * width + k] * matB[k * width + tid_x];
    }
    matC[tid_y * width + tid_x] = accum_val;
  }
#endif
#if 0
  size_t row_idx = blockIdx.x;
  size_t col_idx = threadIdx.x;
  printf("[tid] = %d %d\n", blockIdx.x, threadIdx.x);
  printf("width = %d\n", width);
  if (row_idx < nrow && col_idx < ncol) {
    // float accum_val = 0.0f;
    size_t out_idx = row_idx * ncol + col_idx;
    
    for (size_t k = 0; k < width; ++k) {
      size_t val_a_idx = row_idx * width + k;
      size_t val_b_idx = k * ncol + col_idx;
      printf("[row_col_idx] = %d %d\n", val_a_idx, val_b_idx);
      // float val_a = matA[val_a_idx];
      // float val_a = matA[row_idx * width + k];
      // float val_b = matB[k * width + col_idx];
      // float val_b = matB[val_b_idx];
      // printf("[val] = %f %f\n", val_a, val_b);
      // accum_val += matA[row_idx * width + k] * matB[k * width + col_idx];
      // accum_val += val_a * val_b;
      matC[out_idx] += matA[val_a_idx] * matB[val_b_idx];
    }
    // matC[out_idx] = accum_val;
  }
  // __syncthreads();
#endif
#if 0
  for (size_t i = 0; i < nrow; ++i) {
     for (size_t j = 0; j < ncol; ++j) {
        float accum_val = 0.0;
        for (size_t k = 0; k < width; ++k) {
           accum_val += matA[i * width + k] * matB[k * ncol + j];
        }
        matC[i * ncol + j] = accum_val;
     }
  }
#endif
  size_t row_idx = blockIdx.x;
  size_t col_idx = threadIdx.x;
  if (row_idx < nrow && col_idx < ncol) {
    float accum_val = 0.0;
    for (size_t k = 0; k < width; ++k) {
      size_t val_a_idx = row_idx * width + k;
      size_t val_b_idx = k * ncol + col_idx;
      accum_val += matA[val_a_idx] * matB[val_b_idx];
    }
    size_t out_idx = row_idx * ncol + col_idx;
    matC[out_idx] = accum_val;
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

__global__ void reduce_sum_axis_zero_kernel(int axis0_len,
                                            int nrow, 
                                            int ncol,
                                            const float *input,
                                            float *output) {
#if 0
  for (size_t j = 0; j < nrow; ++j) {
    for (size_t k = 0; k < ncol; ++k) {
      float accum = 0.0;
      for (size_t i = 0; i < axis0_len; ++i) {
        // accum += input[i][j][k];
        accum += input[ (i * nrow + j) * ncol + k ];
      }
      // output[j][k] = accum;
      output[ j * ncol + k ] = accum;
    }
  }
#endif
#if 1
  size_t k = blockIdx.x; // col_idx
  size_t j = blockIdx.y; // row_idx
  if (j < nrow && k < ncol) {
    float accum = 0.0;
    for (size_t i = 0; i < axis0_len; ++i) {
      // accum += input[i][j][k];
      accum += input[ (i * nrow + j) * ncol + k ];
    }
    // output[j][k] = accum;
    output[j * ncol + k] = accum;
  }
#endif
}

__global__ void broadcast_kernel(int new_axis0_len,
                                 int nrow, 
                                 int ncol,
                                 const float *input,
                                 float *output) {
#if 0
  for (size_t i = 0; i < new_axis0_len; ++i) {
    for (size_t j = 0; j < nrow; ++j) {
      for (size_t k = 0; k < ncol; ++k) {
        // output[i][j][k] = input[j][k];
        output[ (i * nrow + j) * ncol + k ] = input[ j * ncol + k ];
      }
    }
  }
#endif
#if 1
  size_t col_idx = blockIdx.x;
  size_t row_idx = blockIdx.y;
  size_t axis0_idx = threadIdx.x;
  if (axis0_idx < new_axis0_len && row_idx < nrow && col_idx < ncol) {
    size_t out_idx = (axis0_idx * nrow + row_idx) * ncol + col_idx;
    size_t in_idx = row_idx * ncol + col_idx;
    output[out_idx] = input[in_idx];
  }
#endif
}

__global__ void relu_kernel(int nrow,
                            int ncol,
                            const float *input,
                            float *output) {
   size_t i = blockIdx.y;
   size_t j = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < nrow && j < ncol) {
     size_t idx = i * ncol + j;
     output[idx] = max(input[idx], 0.0);
   }
#if 0
   for (int i = 0; i < nrow; ++i)
     for (int j = 0; j < ncol; ++j)
       output[i * ncol + j] = max(input[i * ncol + j], 0.0);
#endif
}

__global__ void relu_grad_kernel(int nrow,
                                 int ncol,
                                 const float *input,
                                 const float *in_grad,
                                 float *output) {
#if 1
   size_t i = blockIdx.y;
   size_t j = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < nrow && j < ncol) {
     size_t idx = i * ncol + j;
     output[idx] = input[idx] > 0.0 ? in_grad[idx] : 0.0;
   }
#endif
#if 0
   for (int i = 0; i < nrow; ++i) {
     for (int j = 0; j < ncol; ++j) {
       size_t idx = i * ncol + j;
       output[idx] = input[idx] > 0.0 ? in_grad[idx] : 0.0;
     }
   }
#endif
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
  assert(input->ndim == 2);
  assert(output->ndim == 3);
  int nrow = input->shape[0];
  // Maximum x- or y-dimension of a block = 1024
  // But we need 'nrow' shared memory, and max shared memory is 48KB.
  // Conservatively allow max 16KB shared memory.
  assert(nrow <= 1024 * 4);
  int ncol = input->shape[1];
  int new_axis0_len = output->shape[0];
  const float *input_data = (const float *)input->data;
  float *output_data = (float *)output->data;
  dim3 blocks;
  dim3 threads;
#if 1
  blocks.x = ncol;
  blocks.y = nrow;
  threads.x = new_axis0_len;
#endif
#if 0
  if (nrow * ncol * new_axis0_len <= 1024) {
    threads.x = new_axis0_len;
    threads.y = nrow;
    threads.z = ncol;
  } else {
    blocks.x = ncol;
    blocks.y = nrow;
    threads.x = new_axis0_len;
  }
#endif
  broadcast_kernel<<<blocks, threads>>>(new_axis0_len,
                                        nrow,
                                        ncol, 
                                        input_data, 
                                        output_data);
  return 0;
}

int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  assert(input->ndim == 3);
  assert(output->ndim == 2);
  int nrow = output->shape[0];
  // Maximum x- or y-dimension of a block = 1024
  // But we need 'nrow' shared memory, and max shared memory is 48KB.
  // Conservatively allow max 16KB shared memory.
  assert(nrow <= 1024 * 4);
  int ncol = output->shape[1];
  int axis0_len = input->shape[0];
  const float *input_data = (const float *)input->data;
  float *output_data = (float *)output->data;
  dim3 blocks;
  dim3 threads;
#if 1
  blocks.x = ncol;
  blocks.y = nrow;
  // threads.x = axis0_len;
#endif
#if 0
  if (nrow * ncol * new_axis0_len <= 1024) {
    threads.x = new_axis0_len;
    threads.y = nrow;
    threads.z = ncol;
  } else {
    blocks.x = ncol;
    blocks.y = nrow;
    threads.x = new_axis0_len;
  }
#endif
  reduce_sum_axis_zero_kernel<<<blocks, threads>>>(axis0_len,
                                                   nrow,
                                                   ncol, 
                                                   input_data, 
                                                   output_data);
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
    // blocks.y = (nrow + 1024 - 1) / 1024;
    // blocks.x = ceil(nrow/1024.0);
    // blocks.y = ceil(ncol/1024.0);
    // blocks.x = 3;
    blocks.y = nrow;
    threads.x = 1024;
    // threads.y = 1;
    // threads.x = min(nrow, 1024);
    // threads.y = (nrow + 1023) / 1024;
    // threads.y = 1;
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
  assert(matA->ndim == 2);
  assert(matB->ndim == 2);
  assert(matC->ndim == 2);
  // printf("%d %d %d %d\n", matA->shape[0], matA->shape[1], matB->shape[0], matB->shape[1]);
  // printf("%d %d\n", matC->shape[0], matC->shape[1]);
  assert(matA->shape[1] == matB->shape[0] &&
         matC->shape[0] == matA->shape[0] &&
         matC->shape[1] == matB->shape[1]);
  int nrow = matC->shape[0];
  // Maximum x- or y-dimension of a block = 1024
  // But we need 'nrow' shared memory, and max shared memory is 48KB.
  // Conservatively allow max 16KB shared memory.
  assert(nrow <= 1024 * 4);
  int width = matA->shape[1];
  int ncol = matC->shape[1];
  const float *matA_data = (const float *)matA->data;
  const float *matB_data = (const float *)matB->data;
  float *matC_data = (float *)matC->data;
  dim3 threads;
  dim3 blocks;
#if 0
  if (nrow * ncol <= 1024) {
    threads.x = nrow;
    threads.y = ncol;
  } else {
#endif
    blocks.x = nrow;
    // blocks.y = 1;
    threads.x = min(ncol, 1024);
    // threads.y = 1;
    // blocks.x = (ncol + 1024 - 1) / 1024;
    // blocks.y = nrow;
    // threads.x = min(ncol, 1024); // 1000
    // threads.x = matC->shape[0]; // 500
    // threads.y = 1;
    // threads.x = 1;
    // threads.y = matC->shape[1]; // 1000
    // blocks.x = 1;
    // threads.x = 1;
#if 0
  }
#endif
#if 0
  // printf("here");
  for (size_t i = 0; i < nrow; ++i) {
    for (size_t j = 0; j < width; ++j) {
       size_t idx = i * width + j;
       printf("%d ", idx);
       // printf("%f ", matA_data[idx]);
       // printf("%f ", matA_data[i][j]);
    }
    printf("\n");
  }
  // printf("here1");
  for (size_t i = 0; i < width; ++i) {
    for (size_t j = 0; j < ncol; ++j) {
       size_t idx = i * ncol + j;
       printf("%d ", idx);
       // printf("%f ", matB_data[idx]);
       // printf("%f ", matB_data[i][j]);
    }
    printf("\n");
  }
  // printf("here2");
#endif
  // 1 block, each block with 'threads' number of threads with 'nrow' shared
  // memory size
  matrix__multiply_kernel<<<blocks, threads>>>(matA_data, 
                                               matB_data, 
                                               matC_data,
                                               nrow,
                                               ncol,
                                               width);
  cudaDeviceSynchronize();
  return 0;
}

int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
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
  dim3 blocks;
  dim3 threads;
#if 1
  threads.x = 1024;
  blocks.x = ceil(ncol / 1024);
  blocks.y = nrow;
#endif
  relu_kernel<<<blocks, threads>>>(nrow,
                                   ncol, 
                                   input_data, 
                                   output_data);
  CHECK_GPU_ERR( cudaPeekAtLastError() );
  CHECK_GPU_ERR( cudaDeviceSynchronize() );
  return 0;
}

int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                      DLArrayHandle output) {
  /* TODO: Your code here */
  assert(input->ndim == 2);
  assert(in_grad->ndim == 2);
  assert(output->ndim == 2);
  int nrow = input->shape[0];
  // Maximum x- or y-dimension of a block = 1024
  // But we need 'nrow' shared memory, and max shared memory is 48KB.
  // Conservatively allow max 16KB shared memory.
  assert(nrow <= 1024 * 4);
  int ncol = input->shape[1];
  const float *input_data = (const float *)input->data;
  const float *in_grad_data = (const float *)in_grad->data;
  float *output_data = (float *)output->data;
  dim3 blocks;
  dim3 threads;
#if 1
  threads.x = 1024;
  blocks.x = (ncol + 1024 - 1) / 1024;
  blocks.y = nrow;
#endif
#if 0
  threads.x = 2;
  // blocks.x = 3;
  blocks.x = (ncol + 2 - 1 / 2);
  blocks.y = nrow;
#endif

  relu_grad_kernel<<<blocks, threads>>>(nrow,
                                        ncol, 
                                        input_data, 
                                        in_grad_data,
                                        output_data);
  CHECK_GPU_ERR( cudaPeekAtLastError() );
  CHECK_GPU_ERR( cudaDeviceSynchronize() );
  return 0;
}

int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output) {
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
  dim3 blocks;
  dim3 threads;
#if 0
  threads.x = 1024;
  blocks.x = ceil(ncol / 1024);
  blocks.y = nrow;
#endif
  softmax_kernel<<<blocks, threads>>>(nrow,
                                      ncol, 
                                      input_data, 
                                      output_data);
  CHECK_GPU_ERR( cudaPeekAtLastError() );
  CHECK_GPU_ERR( cudaDeviceSynchronize() );
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
