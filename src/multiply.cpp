#include <bits/stdc++.h>
#include <immintrin.h>
#include <sys/time.h>

#include <cstddef>
void simple_multiply(float *mat1, float *mat2, std::ptrdiff_t N,
                     std::ptrdiff_t M, std::ptrdiff_t O, float *res) {
  for (std::ptrdiff_t i = 0; i < N; i++) {
    for (std::ptrdiff_t j = 0; j < M; j++) {
      for (std::ptrdiff_t k = 0; k < O; k++)
        res[i * M + j] += mat1[i * O + k] * mat2[k * M + j];
    }
  }
}

void sum(const float *A, std::ptrdiff_t A_row, const float *B,
         std::ptrdiff_t B_row, float *C, std::ptrdiff_t C_row, std::ptrdiff_t N,
         std::ptrdiff_t M) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; j += 8) {
      __m256 A_vector = _mm256_loadu_ps(&A[i * A_row + j]);
      __m256 B_vector = _mm256_loadu_ps(&B[i * B_row + j]);
      _mm256_storeu_ps(&C[i * C_row + j], _mm256_add_ps(A_vector, B_vector));
    }
  }
}

void sub(const float *A, std::ptrdiff_t A_row, const float *B,
         std::ptrdiff_t B_row, float *C, std::ptrdiff_t C_row, std::ptrdiff_t N,
         std::ptrdiff_t M) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; j += 8) {
      __m256 A_vector = _mm256_loadu_ps(&A[i * A_row + j]);
      __m256 B_vector = _mm256_loadu_ps(&B[i * B_row + j]);
      _mm256_storeu_ps(&C[i * C_row + j], _mm256_sub_ps(A_vector, B_vector));
    }
  }
}

void smallMultyply(std::ptrdiff_t N, std::ptrdiff_t M, std::ptrdiff_t O,
                   const float *A, std::ptrdiff_t A_row, const float *B,
                   std::ptrdiff_t B_row, float *C, std::ptrdiff_t C_row) {
  memset(C, 0, sizeof(float) * N * M);
  for (int i = 0; i < N; ++i) {
    for (int k = 0; k < O; ++k) {
      __m256 A_number = _mm256_broadcast_ss(&A[i * A_row + k]);
      for (int j = 0; j < M; j += 8) {
        __m256 B_vector = _mm256_loadu_ps(&B[k * B_row + j]);
        __m256 mulResult = _mm256_mul_ps(A_number, B_vector);
        __m256 resultMatrix = _mm256_loadu_ps(&C[i * C_row + j]);
        _mm256_storeu_ps(&C[i * C_row + j],
                         _mm256_add_ps(mulResult, resultMatrix));
      }
    }
  }
}

void strassenAlg(std::ptrdiff_t N, std::ptrdiff_t M, std::ptrdiff_t O,
                 const float *A, const std::ptrdiff_t A_row, const float *B,
                 std::ptrdiff_t B_row, float *C, std::ptrdiff_t C_row) {
  if (N == 1 || M == 1 || O == 1 || N * M * O < 128 * 128 * 128) {
    smallMultyply(N, M, O, A, A_row, B, B_row, C, C_row);
    return;
  }

  std::ptrdiff_t N_2 = N / 2;
  std::ptrdiff_t M_2 = M / 2;
  std::ptrdiff_t O_2 = O / 2;

  // A = N * O
  const float *A11 = &A[0];
  const float *A12 = &A[O_2];
  const float *A21 = &A[N_2 * A_row];
  const float *A22 = &A[N_2 * A_row + O_2];

  // B = O * M
  const float *B11 = &B[0];
  const float *B12 = &B[M_2];
  const float *B21 = &B[O_2 * B_row];
  const float *B22 = &B[O_2 * B_row + M_2];

  // C = N * M
  float *C11 = &C[0];
  float *C12 = &C[M_2];
  float *C21 = &C[N_2 * C_row];
  float *C22 = &C[N_2 * C_row + M_2];

  float *S1 = (float *)malloc(sizeof(float) * N_2 * O_2);
  float *S2 = (float *)malloc(sizeof(float) * O_2 * M_2);
  float *S3 = (float *)malloc(sizeof(float) * N_2 * O_2);
  float *S4 = (float *)malloc(sizeof(float) * O_2 * M_2);
  float *S5 = (float *)malloc(sizeof(float) * O_2 * M_2);
  float *S6 = (float *)malloc(sizeof(float) * N_2 * O_2);
  float *S7 = (float *)malloc(sizeof(float) * N_2 * O_2);
  float *S8 = (float *)malloc(sizeof(float) * O_2 * M_2);
  float *S9 = (float *)malloc(sizeof(float) * N_2 * O_2);
  float *S10 = (float *)malloc(sizeof(float) * O_2 * M_2);

  float *P1 = (float *)malloc(sizeof(float) * N_2 * M_2);
  float *P2 = (float *)malloc(sizeof(float) * N_2 * M_2);
  float *P3 = (float *)malloc(sizeof(float) * N_2 * M_2);
  float *P4 = (float *)malloc(sizeof(float) * N_2 * M_2);
  float *P5 = (float *)malloc(sizeof(float) * N_2 * M_2);
  float *P6 = (float *)malloc(sizeof(float) * N_2 * M_2);
  float *P7 = (float *)malloc(sizeof(float) * N_2 * M_2);

  sum(A11, A_row, A22, A_row, S1, O_2, N_2, O_2);
  sum(B11, B_row, B22, B_row, S2, M_2, O_2, M_2);
  strassenAlg(N_2, M_2, O_2, S1, O_2, S2, M_2, P1, M_2);
  sum(A21, A_row, A22, A_row, S3, O_2, N_2, O_2);
  strassenAlg(N_2, M_2, O_2, S3, O_2, B11, B_row, P2, M_2);
  sub(B12, B_row, B22, B_row, S4, M_2, O_2, M_2);
  strassenAlg(N_2, M_2, O_2, A11, A_row, S4, M_2, P3, M_2);
  sub(B21, B_row, B11, B_row, S5, M_2, O_2, M_2);
  strassenAlg(N_2, M_2, O_2, A22, A_row, S5, M_2, P4, M_2);
  sum(A11, A_row, A12, A_row, S6, O_2, N_2, O_2);
  strassenAlg(N_2, M_2, O_2, S6, O_2, B22, B_row, P5, M_2);
  sub(A21, A_row, A11, A_row, S7, O_2, N_2, O_2);
  sum(B11, B_row, B12, B_row, S8, M_2, O_2, M_2);
  strassenAlg(N_2, M_2, O_2, S7, O_2, S8, M_2, P6, M_2);
  sub(A12, A_row, A22, A_row, S9, O_2, N_2, O_2);
  sum(B21, B_row, B22, B_row, S10, M_2, O_2, M_2);
  strassenAlg(N_2, M_2, O_2, S9, O_2, S10, M_2, P7, M_2);

  for (std::ptrdiff_t i = 0; i < N_2; ++i) {
    int ind = i * C_row, ind2 = i * M_2;
    for (std::ptrdiff_t j = 0; j < M_2; ++j) {
      C11[ind + j] = P1[ind2 + j] + P4[ind2 + j] - P5[ind2 + j] + P7[ind2 + j];
      C12[ind + j] = P3[ind2 + j] + P5[ind2 + j];
      C21[ind + j] = P2[ind2 + j] + P4[ind2 + j];
      C22[ind + j] = P1[ind2 + j] - P2[ind2 + j] + P3[ind2 + j] + P6[ind2 + j];
    }
  }
  free(S1);
  free(S2);
  free(S3);
  free(S4);
  free(S5);
  free(S6);
  free(S7);
  free(S8);
  free(S9);
  free(S10);
  free(P1);
  free(P2);
  free(P3);
  free(P4);
  free(P5);
  free(P6);
  free(P7);
}

void multiply(float *mat1, float *mat2, std::ptrdiff_t N, std::ptrdiff_t M,
              std::ptrdiff_t O, float *res) {
  // printf("Executing ");
  // fflush(stdout);
  // struct timeval before, after;
  // gettimeofday(&before, NULL);
  strassenAlg(N, M, O, mat1, O, mat2, M, res, M);
  // gettimeofday(&after, NULL);

  // float tdiff =
  //     after.tv_sec - before.tv_sec + (1e-6) * (after.tv_usec -
  //     before.tv_usec);
  // printf("\nsecs:%11.6f\n", tdiff);
}