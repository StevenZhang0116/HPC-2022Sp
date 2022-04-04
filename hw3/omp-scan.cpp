#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <iostream>

using namespace std;

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  int nt; long ps;
  # pragma omp parallel
  {
    int tid = omp_get_thread_num();
    nt = 4; // set the number of threads
            // nt = 4 = omp_get_num_threads()
    if (tid == 0) cout << nt << endl;
    ps = n/nt;
    long start1 = ps * tid + 1;
    long end1 = min(ps * (tid + 1) + 1, n);
    for (long i = start1; i < end1; ++i) {
      prefix_sum[i] = prefix_sum[i-1] + A[i-1];
    }
  }
  for (int j = 1; j < nt; j++) {
    long start2 = ps * (long)j + 1;
    long end2 = min(ps * ((long)j + 1) + 1, n);
    for (long i = start2; i < end2; ++i) {
      prefix_sum[i] += prefix_sum[start2 - 1];
    }
  }
}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
