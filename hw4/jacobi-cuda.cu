#include <cmath>
#include <iostream>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "utils.h"

using namespace std;

#define BLOCK_DIM 16
#define BLOCK_DIM_I 14

// From HW2
double residual(int N, double *u, double *f){
    double h=1.0/(N+1.0);
    double count=0;
    double res=0;
    #pragma omp parallel for shared(u,f) reduction(+:count)
    for(int j=1;j<=N;j++){
        for(int i=1;i<=N;i++){
            res=(-u[(N+2)*j+i-1]-u[(N+2)*(j-1)+i]-u[(N+2)*j+i+1]-u[(N+2)*(j+1)+i]+4.0*u[(N+2)*j+i])/(h*h)-f[(N+2)*j+i];
            count+=res*res;
        }
    }
    count=sqrt(count);
    return count;
}

// From HW2
double jacobi_cpu(int N, double *u, double *f, int maxiteration, int numt){
    #if defined(_OPENMP)
    omp_set_num_threads(numt);
    cout << "use threads: " << numt << endl;
    #endif
    double h=1.0/(N+1.0);
    double res=0.0;
    double tol=1e-8;
    double count=0.0;
    int iter_count=0;
    double *copy = (double*) malloc((N+2)*(N+2) * sizeof(double));
    double res_init=residual(N,u,f);
    cout << "Initial Residual:" << res_init << endl;
    count=tol+1.0;
    while(count>tol){
        #pragma omp parallel shared(copy,u)
        {
            #pragma omp for
            for(int j=1;j<=N;j++){
                for(int i=1;i<=N;i++){
                    copy[(N+2)*j+i]=0.25*(h*h*f[(N+2)*j+i]+u[(N+2)*j+i-1]+u[(N+2)*(j-1)+i]+u[(N+2)*j+i+1]+u[(N+2)*(j+1)+i]);
                }
            }
            #pragma omp for
            for(int j=1;j<=N;j++){
                for(int i=1;i<=N;i++){
                    u[(N+2)*j+i]=copy[(N+2)*j+i];
                }
            }
        }
        res=residual(N,u,f);
        count=res/res_init;
        iter_count++;
        if(iter_count>maxiteration){
            cout << "beyond max iteration" << endl;
            cout << "Remaining res:" << count << endl;
            break;
        }
    }
    cout << "Remaining res:" << count << endl;
    free(copy);
    return count;
}

__global__ void jacobi_kernel(int N, double h, double *unew, double *u, double *f){
    __shared__ double smem[BLOCK_DIM][BLOCK_DIM];
    if(blockIdx.x*BLOCK_DIM_I+threadIdx.x<N+2 && blockIdx.y*BLOCK_DIM_I+threadIdx.y<N+2){
        smem[threadIdx.x][threadIdx.y] = u[(blockIdx.y*BLOCK_DIM_I+threadIdx.y)*(N+2)+blockIdx.x*BLOCK_DIM_I+threadIdx.x];
    }
    __syncthreads();
    if(threadIdx.x<=BLOCK_DIM_I && threadIdx.x>=1 && threadIdx.y<=BLOCK_DIM_I && threadIdx.y>=1){
        if(blockIdx.x*BLOCK_DIM_I+threadIdx.x<N+1 && blockIdx.x*BLOCK_DIM_I+threadIdx.x>0
            && blockIdx.y*BLOCK_DIM_I+threadIdx.y<N+1 && blockIdx.y*BLOCK_DIM_I+threadIdx.y>0){
                unew[(blockIdx.y*BLOCK_DIM_I+threadIdx.y)*(N+2)+blockIdx.x*BLOCK_DIM_I+threadIdx.x] = 0.25*(h*h* 
                f[(blockIdx.y*BLOCK_DIM_I+threadIdx.y)*(N+2)+blockIdx.x*BLOCK_DIM_I+threadIdx.x]+smem[threadIdx.x-1][threadIdx.y]
                +smem[threadIdx.x+1][threadIdx.y]+smem[threadIdx.x][threadIdx.y-1]+smem[threadIdx.x][threadIdx.y+1]);
        }
    }
}

double jacobi_gpu(int N, double *u, double *f, int maxiteration){
    double h=1.0/(N+1.0);
    double res=0.0;
    double tol=1e-8;
    double count=0.0;
    int iter_count=0;

    double *unew, *u1, *f1;
    cudaMalloc(&u1, (N+2)*(N+2)*sizeof(double));
    cudaMalloc(&f1, (N+2)*(N+2)*sizeof(double));
    cudaMemcpy(u1, u, (N+2)*(N+2)*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(f1, f, (N+2)*(N+2)*sizeof(double),cudaMemcpyHostToDevice);

    cudaMalloc(&unew, (N+2)*(N+2)*sizeof(double));
    cudaMemcpy(unew, u1, (N+2)*(N+2)*sizeof(double),cudaMemcpyDeviceToDevice);

    double res_init=residual(N,u,f);
    cout << "Initial Residual:" << res_init << endl;
    count=tol+1.0;
    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim((N-1)/(BLOCK_DIM_I)+1, (N-1)/(BLOCK_DIM_I)+1);
    while (count>tol){
        jacobi_kernel<<<gridDim,blockDim>>>(N, h, unew, u1, f1);
        cudaMemcpy(u1,unew,(N+2)*(N+2)*sizeof(double),cudaMemcpyDeviceToDevice);
        cudaMemcpy(u,u1,(N+2)*(N+2)*sizeof(double),cudaMemcpyDeviceToHost);
        res=residual(N,u,f);
        count=res/res_init;

        iter_count++;
        if (iter_count>maxiteration){
        cout << "beyond max iteration" << endl;
        cout << "Remaining res:" << count << endl;
        break;
        }
    }
    cout << "Remaining res:" << count << endl;
    cudaFree(unew);
    return count;
}


int main(int argc, char **argv){
    int N = 20;
    int numt = 4;
    int maxiteration = 1000000;
    double r1, r2;
    double *u = (double*) malloc ((N+2)*(N+2)*sizeof(double));
    double *f = (double*) malloc ((N+2)*(N+2)*sizeof(double));
    memset(u,0,(N+2)*(N+2)*sizeof(double));
    memset(f,0,(N+2)*(N+2)*sizeof(double));
    for(int i = 0; i < (N+2)*(N+2); ++i){
        f[i] = 1.0;
    }
    Timer t;
    cout << "========CPU========" << endl;
    t.tic();
    r1 = jacobi_cpu(N,u,f,maxiteration,numt);
    printf("CPU Bandwidth = %f GB/s\n", maxiteration*10*(N+2)*(N+2)*sizeof(double) / (t.toc())/1e9);
    cout << "CPU Time:" << t.toc() << "s" << endl;
    

    memset(u,0,(N+2)*(N+2)*sizeof(double));
    cout << "========GPU========" << endl;
    t.tic();
    r2 = jacobi_gpu(N,u,f,maxiteration);
    printf("GPU Bandwidth = %f GB/s\n", maxiteration*10*(N+2)*(N+2)*sizeof(double) / (t.toc())/1e9);
    cout << "GPU Time:" << t.toc() << "s" << endl;

    cout << "========Result Comparison========" << endl;
    cout << "CPU-GPU difference:" << r1-r2 << endl;

    free(u);
    free(f);
    return 0;
}