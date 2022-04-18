// Matrix - Vector Multiplication
// Reference: https://github.com/NYU-HPC19/lecture8/blob/master; Professor Stadler, NYU, who previously taught this course

#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>
#include <stdlib.h>
#include <time.h>
#include <iostream>

// void product_cpu_cal2(double* sum_ptr, const double* a, const double* b, long M, long N){
//     double sum = 0;
//     for(long j = 0; j < M; ++j){
//         sum = 0;
//         #pragma omp parallel for schedule(static) reduction(+:sum)
//         for(long i = 0; i < N; ++i){
//             sum += a[j*M+i] * b[i];
//         }
//         sum_ptr[j] = sum;
//     }
// }

void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

double error_func(double* x, double* y, int size){
    double total_val = 0.0;
    for(int i = 0; i < size; ++i) total_val = max(fabs(x[i] - y[i]), total_val);
    return total_val;
}

// From previous HWs
void MMult(long M, long N, double* A, double* x, double* c){
    #pragma omp parallel for
    for(long i = 0; i < M; ++i){
        for(long j = 0; j < N; ++j){
            c[i] += A[i*M+j] * x[j];
        }
    }
}
#define BLOCK_SIZE 1024

__global__ void reduction_kernel(double* sum, const double* a, long N){
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < N) smem[threadIdx.x] = a[idx];
  else smem[threadIdx.x] = 0;

  __syncthreads();
  if(threadIdx.x < 512) smem[threadIdx.x] += smem[threadIdx.x + 512];
  __syncthreads();
  if(threadIdx.x < 256) smem[threadIdx.x] += smem[threadIdx.x + 256];
  __syncthreads();
  if(threadIdx.x < 128) smem[threadIdx.x] += smem[threadIdx.x + 128];
  __syncthreads();
  if(threadIdx.x <  64) smem[threadIdx.x] += smem[threadIdx.x + 64];
  __syncthreads();
  if(threadIdx.x < 32){
      smem[threadIdx.x] += smem[threadIdx.x + 32];
      __syncwarp();
      smem[threadIdx.x] += smem[threadIdx.x + 16];
      __syncwarp();
      smem[threadIdx.x] += smem[threadIdx.x + 8];
      __syncwarp();
      smem[threadIdx.x] += smem[threadIdx.x + 4];
      __syncwarp();
      smem[threadIdx.x] += smem[threadIdx.x + 2];
      __syncwarp();
      if(threadIdx.x == 0) sum[blockIdx.x] = smem[0] + smem[1];
  }
}

__global__ void reduction_product(double* sum, const double* a, const double* b, long N){
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < N) smem[threadIdx.x] = a[idx] * b[idx];
  else smem[threadIdx.x] = 0;

  __syncthreads();
  if(threadIdx.x < 512) smem[threadIdx.x] += smem[threadIdx.x + 512];
  __syncthreads();
  if(threadIdx.x < 256) smem[threadIdx.x] += smem[threadIdx.x + 256];
  __syncthreads();
  if(threadIdx.x < 128) smem[threadIdx.x] += smem[threadIdx.x + 128];
  __syncthreads();
  if(threadIdx.x < 64) smem[threadIdx.x] += smem[threadIdx.x + 64];
  __syncthreads();
  if(threadIdx.x < 32){
      smem[threadIdx.x] += smem[threadIdx.x + 32];
      __syncwarp();
      smem[threadIdx.x] += smem[threadIdx.x + 16];
      __syncwarp();
      smem[threadIdx.x] += smem[threadIdx.x + 8];
      __syncwarp();
      smem[threadIdx.x] += smem[threadIdx.x + 4];
      __syncwarp();
      smem[threadIdx.x] += smem[threadIdx.x + 2];
      __syncwarp();
      if(threadIdx.x == 0) sum[blockIdx.x] = smem[0] + smem[1];
  }
}

int main(){
    long N = (1UL << 10);
    long M = (1UL << 10);

    double *a, *b;
    cudaMallocHost((void**)&a, N * M * sizeof(double));
    cudaMallocHost((void**)&b, N * sizeof(double));
    #pragma omp parallel for schedule(static)
    for(long i = 0; i < N; ++i){ // add more randomness here
        b[i] = ((double)rand())/RAND_MAX;
    }
    #pragma omp parallel for schedule(static)
    for(long i = 0; i < N*M; ++i){ 
        a[i] = ((double)rand())/RAND_MAX;
    }

    double *sum_ref, *sum;
    sum_ref = (double *)malloc(M*sizeof(double));
    sum = (double *)malloc(M*sizeof(double));
    for(int i = 0; i < M; ++i){
        sum_ref[i] = 0;
        sum[i] = 0;
    }

    double tt = omp_get_wtime();
    MMult(M, N, a, b, sum_ref);
    printf("CPU Bandwidth = %f GB/s\n", 2*M*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

    double *x_d, *y_d, *z_d;
    cudaMalloc(&x_d, N*sizeof(double));
    cudaMalloc(&y_d, N*sizeof(double));
    long N_work = 1;
    for (long i = (N+BLOCK_SIZE-1)/(BLOCK_SIZE); i > 1; i = (i+BLOCK_SIZE-1)/(BLOCK_SIZE)) N_work += i;
    cudaMalloc(&z_d, N_work*sizeof(double));
    cudaMemcpyAsync(y_d, b, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    tt = omp_get_wtime();

    for(int i = 0; i < M; ++i){
        double* sum_d;
        cudaMalloc(&sum_d, N_work*sizeof(double));
        cudaMemcpyAsync(x_d, &(a[i*N]), N*sizeof(double), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        long Nb = (N+BLOCK_SIZE-1)/(BLOCK_SIZE);
        reduction_product<<<Nb,BLOCK_SIZE>>>(sum_d, x_d, y_d, N);
        while(Nb > 1){
            long N1 = Nb;
            Nb = (Nb+BLOCK_SIZE-1)/(BLOCK_SIZE);
            reduction_kernel<<<Nb,BLOCK_SIZE>>>(sum_d + N1, sum_d, N1);
            sum_d += N1;
        }
        cudaMemcpyAsync(&(sum[i]), sum_d, sizeof(double), cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();
    printf("GPU Bandwidth = %f GB/s\n", 2*M*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);
    printf("Error = %f\n", error_func(sum,sum_ref,M));

    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);
    free(sum);
    free(sum_ref);

    return 0;
}

