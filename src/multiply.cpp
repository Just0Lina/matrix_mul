#include <bits/stdc++.h>
#include <immintrin.h>
#include <sys/time.h>

#include <cstddef>
#include <iostream>
void simple_multiply(float *mat1, float *mat2, std::ptrdiff_t N,
                     std::ptrdiff_t M, std::ptrdiff_t O, float *res) {
  for (std::ptrdiff_t i = 0; i < N; i++) {
    for (std::ptrdiff_t j = 0; j < M; j++) {
      for (std::ptrdiff_t k = 0; k < O; k++)
        res[i * M + j] += mat1[i * O + k] * mat2[k * M + j];
    }
  }
}

void multiply2(double *A, double *B, long N, long M, long K, double *C) {
  printf("Executing ");
  fflush(stdout);
  int bSize = 64;
  struct timeval before, after;
  gettimeofday(&before, NULL);
  for (int jk = 0; jk < N / bSize; ++jk) {
    for (int ik = 0; ik < M / bSize; ++ik) {
      for (int j = jk * bSize; j < jk * bSize + bSize; ++j) {
        double *c = C + j * K;
        for (int k = ik * bSize; k < ik * bSize + bSize; ++k) {
          double *b = B + k * K;
          __m256d a = _mm256_set1_pd(A[j * M + k]);
          for (int i = 0; i < K; i += 4) {
            _mm256_storeu_pd(c + i + 0,
                             _mm256_fmadd_pd(a, _mm256_loadu_pd(b + i + 0),
                                             _mm256_loadu_pd(c + i + 0)));
          }
        }
      }
    }
  }
  gettimeofday(&after, NULL);

  double tdiff =
      after.tv_sec - before.tv_sec + (1e-6) * (after.tv_usec - before.tv_usec);
  printf("\nsecs:%11.6f\n", tdiff);
}

void multiply(float *A, float *B, long N, long M, long K, float *C) {
  // printf("Executing ");
  // fflush(stdout);
  // struct timeval before, after;
  // gettimeofday(&before, NULL);
  int bSize = 64;
  for (int jk = 0; jk < N / bSize; ++jk) {
    for (int ik = 0; ik < K / bSize; ++ik) {
      for (int j = jk * bSize; j < jk * bSize + bSize; ++j) {
        float *c = C + j * M;
        for (int k = ik * bSize; k < ik * bSize + bSize; ++k) {
          float *b = B + k * M;
          __m256 a = _mm256_set1_ps(A[j * K + k]);
          for (int i = 0; i < M; i += 8) {
            _mm256_storeu_ps(c + i + 0,
                             _mm256_fmadd_ps(a, _mm256_loadu_ps(b + i + 0),
                                             _mm256_loadu_ps(c + i + 0)));
          }
        }
      }
    }
  }
  // gettimeofday(&after, NULL);

  // float tdiff =
  //     after.tv_sec - before.tv_sec + (1e-6) * (after.tv_usec -
  //     before.tv_usec);
  // printf("\nsecs:%11.6f\n", tdiff);
}