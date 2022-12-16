#include <bits/stdc++.h>
#include <immintrin.h>
#include <sys/time.h>
// #include <omp.h>
unsigned long long GRAIN = 512 * 512 * 512;
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

// void multiply(double *A, double *B, long M, long N, long K, double *C) {
//   printf("Executing ");
//   fflush(stdout);
//   struct timeval before, after;
//   gettimeofday(&before, NULL);

//   for (int i = 0; i < M; ++i) {
//     double *c = C + i * N;
//     for (int k = 0; k < K; ++k) {
//       const double *b = B + k * N;
//       __m256d a = _mm256_set1_pd(A[i * K + k]);
//       for (int j = 0; j < N; j += 4) {
//         _mm256_storeu_pd(c + j + 0,
//                          _mm256_fmadd_pd(a, _mm256_loadu_pd(b + j + 0),
//                                          _mm256_loadu_pd(c + j + 0)));
//         // _mm256_storeu_pd(c + j + 4,
//         //                  _mm256_fmadd_pd(a, _mm256_loadu_pd(b + j + 4),
//         //                                  _mm256_loadu_pd(c + j + 4)));
//       }
//     }
//   }
//   gettimeofday(&after, NULL);

//   double tdiff =
//       after.tv_sec - before.tv_sec + (1e-6) * (after.tv_usec -
//       before.tv_usec);
//   printf("\nsecs:%11.6f\n", tdiff);
// }

// void multiply(double *A, double *B, long N, long M, long O, double *C) {
//   printf("Executing ");
//   fflush(stdout);
//   struct timeval before, after;
//   gettimeofday(&before, NULL);

//   int i, j, k;
// #pragma omp parallel for private(i, j, k) shared(A, B, C)
//   for (i = 0; i < N; ++i) {
//     int c = i * O, aa = i * M;
//     for (k = 0; k < O; ++k) {
//       double a = A[aa + k];
//       int b = k * O;
//       for (j = 0; j < M; ++j) {
//         C[c + j] += a * B[b + j];
//       }
//     }
//   }
//   gettimeofday(&after, NULL);

//   double tdiff =
//       after.tv_sec - before.tv_sec + (1e-6) * (after.tv_usec -
//       before.tv_usec);
//   printf("\nsecs:%11.6f\n", tdiff);
// }

// #include <stdio.h>
// #include <string.h>

// void Rec_Mult(double *C, double *A, double *B, int n, int rowsize) {
//   if (n == 2) {
//     const int d11 = 0;
//     const int d12 = 1;
//     const int d21 = rowsize;
//     const int d22 = rowsize + 1;

//     C[0] += A[0] * B[0] + A[1] * B[d21];
//     C[1] += A[0] * B[1] + A[1] * B[d22];
//     C[d21] += A[d21] * B[0] + A[d22] * B[d21];
//     C[d22] += A[d21] * B[1] + A[d22] * B[d22];
//   } else {
//     const int d11 = 0;
//     const int d12 = n / 2;
//     const int d21 = (n / 2) * rowsize;
//     const int d22 = (n / 2) * (rowsize + 1);

//     // C11 += A11 * B11
//     Rec_Mult(C + d11, A + d11, B + d11, d12, rowsize);
//     // C11 += A12 * B21
//     Rec_Mult(C + d11, A + d12, B + d21, d12, rowsize);

//     // C12 += A11 * B12
//     Rec_Mult(C + d12, A + d11, B + d12, d12, rowsize);
//     // C12 += A12 * B22
//     Rec_Mult(C + d12, A + d12, B + d22, d12, rowsize);

//     // C21 += A21 * B11
//     Rec_Mult(C + d21, A + d21, B + d11, d12, rowsize);
//     // C21 += A22 * B21
//     Rec_Mult(C + d21, A + d22, B + d21, n / 2, rowsize);

//     // C22 += A21 * B12
//     Rec_Mult(C + d22, A + d21, B + d12, d12, rowsize);
//     // C22 += A22 * B22
//     Rec_Mult(C + d22, A + d22, B + d22, d12, rowsize);
//   }
// }

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
  printf("Executing ");
  fflush(stdout);
  struct timeval before, after;
  gettimeofday(&before, NULL);
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
  gettimeofday(&after, NULL);

  float tdiff =
      after.tv_sec - before.tv_sec + (1e-6) * (after.tv_usec - before.tv_usec);
  printf("\nsecs:%11.6f\n", tdiff);
}