#include "utils.h"
#include <cmath>
#if defined(_OPENMP)
#include <omp.h>
#endif

using namespace std;

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

void jacobi(int N, double *u, double *f, int maxiteration, int numt){
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
            break;
        }
    }
    free(copy);
}

int main(int argc, char **argv){
    int N=200;
    int num_threads=2;
    int maxiteration=10000;
    double *u=(double*) malloc ((N+2)*(N+2)*sizeof(double));
    double *f=(double*) malloc ((N+2)*(N+2)*sizeof(double));
    #pragma omp for
    for(int i=0;i<(N+2)*(N+2);i++){
        u[i]=0.0;
        f[i]=1.0;
    }
    Timer t;
    t.tic();
    jacobi(N,u,f,maxiteration,num_threads);
    cout << "time:" << t.toc() << endl;
    free(u);
    free(f);
    return 0;
}