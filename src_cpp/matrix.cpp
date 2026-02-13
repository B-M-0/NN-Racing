#include "matrix.h"
#include <immintrin.h>
#include <thread>
#include <vector>

Matrix::Matrix(int m, int n, std::vector<float> x, Rank rank) {
  this->rows = m;
  this->cols = n;
  this->elements = x;
  this->rank = rank;
}

Matrix::Matrix(int m, int n, Rank rank) {
  this->rows = m;
  this->cols = n;
  this->elements = std::vector<float>(m * n);
  this->rank = rank;
}

int Matrix::get_rows() const { return this->rows; }

int Matrix::get_columns() const { return this->cols; }

void Matrix::transpose() {
  if (this->rank == row)
    this->rank = column;
  else
    this->rank = row;
}

float Matrix::operator()(int idx_r, int idx_c) const {
  if (this->rank == row)
    return elements[idx_r * cols + idx_c];
  else
    return elements[idx_c * rows + idx_r];
}

float &Matrix::operator()(int idx_r, int idx_c) {
  if (this->rank == row)
    return elements[idx_r * cols + idx_c];
  else
    return elements[idx_c * rows + idx_r];
}

float dot(const std::vector<float> &a, const std::vector<float> &b) {
  float sum = 0.0f;
  for (size_t i = 0; i < a.size(); i++)
    sum += a[i] * b[i];
  return sum;
}

Matrix Matrix::operator*(const Matrix &A) const {
  // Result dimensions: (this->rows) x (A.cols)
  // The result of (Row-Major) * (Any) is naturally Row-Major in this
  // implementation logic.
  Matrix output(this->rows, A.cols, row);

  // Determine number of threads
  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0)
    num_threads = 2; // Fallback
  // Don't use more threads than rows
  if (num_threads > (unsigned int)this->rows)
    num_threads = this->rows;

  auto worker = [&](int start_row, int end_row) {
    // Fast Path: Row-Major * Column-Major
    // Optimized with L2 Tiling and Register Blocking (2x6)
    if (this->rank == row && A.rank == column) {
      const float *this_data = this->elements.data();
      const float *A_data = A.elements.data();
      float *out_data = output.elements.data();

      // Constants for Tiling
      const int BLOCK_COL_B =
          256; // Block size for B columns to fit in L3 (Optimistic)

      // Iterate over blocks of columns of B (L2 Cache Tiling)
      // This ensures we reuse the rows of A against a resident block of B
      for (int jc = 0; jc < A.cols; jc += BLOCK_COL_B) {
        int j_block_end = std::min(jc + BLOCK_COL_B, A.cols);

        // Iterate over rows of A (assigned to this thread)
        // We unroll by 2 for Register Blocking (2 rows of A at a time)
        int i = start_row;
        for (; i <= end_row - 2; i += 2) {
          const float *row_ptr0 = this_data + (i * this->cols);
          const float *row_ptr1 = this_data + ((i + 1) * this->cols);
          float *out_ptr0 = out_data + (i * output.cols);
          float *out_ptr1 = out_data + ((i + 1) * output.cols);

          int j = jc;
          // Register Block: 2 Rows of A x 6 Cols of B
          // 12 Accumulators
          for (; j <= j_block_end - 6; j += 6) {
            __m256 sum00 = _mm256_setzero_ps();
            __m256 sum01 = _mm256_setzero_ps();
            __m256 sum02 = _mm256_setzero_ps();
            __m256 sum03 = _mm256_setzero_ps();
            __m256 sum04 = _mm256_setzero_ps();
            __m256 sum05 = _mm256_setzero_ps();

            __m256 sum10 = _mm256_setzero_ps();
            __m256 sum11 = _mm256_setzero_ps();
            __m256 sum12 = _mm256_setzero_ps();
            __m256 sum13 = _mm256_setzero_ps();
            __m256 sum14 = _mm256_setzero_ps();
            __m256 sum15 = _mm256_setzero_ps();

            const float *col_ptr0 = A_data + (j * A.rows);
            const float *col_ptr1 = A_data + ((j + 1) * A.rows);
            const float *col_ptr2 = A_data + ((j + 2) * A.rows);
            const float *col_ptr3 = A_data + ((j + 3) * A.rows);
            const float *col_ptr4 = A_data + ((j + 4) * A.rows);
            const float *col_ptr5 = A_data + ((j + 5) * A.rows);

            int k = 0;
            // Inner K loop unrolled
            for (; k <= this->cols - 8; k += 8) {
              __m256 a0 = _mm256_loadu_ps(row_ptr0 + k);
              __m256 a1 = _mm256_loadu_ps(row_ptr1 + k);

              __m256 b0 = _mm256_loadu_ps(col_ptr0 + k);
              sum00 = _mm256_fmadd_ps(a0, b0, sum00);
              sum10 = _mm256_fmadd_ps(a1, b0, sum10);

              __m256 b1 = _mm256_loadu_ps(col_ptr1 + k);
              sum01 = _mm256_fmadd_ps(a0, b1, sum01);
              sum11 = _mm256_fmadd_ps(a1, b1, sum11);

              __m256 b2 = _mm256_loadu_ps(col_ptr2 + k);
              sum02 = _mm256_fmadd_ps(a0, b2, sum02);
              sum12 = _mm256_fmadd_ps(a1, b2, sum12);

              __m256 b3 = _mm256_loadu_ps(col_ptr3 + k);
              sum03 = _mm256_fmadd_ps(a0, b3, sum03);
              sum13 = _mm256_fmadd_ps(a1, b3, sum13);

              __m256 b4 = _mm256_loadu_ps(col_ptr4 + k);
              sum04 = _mm256_fmadd_ps(a0, b4, sum04);
              sum14 = _mm256_fmadd_ps(a1, b4, sum14);

              __m256 b5 = _mm256_loadu_ps(col_ptr5 + k);
              sum05 = _mm256_fmadd_ps(a0, b5, sum05);
              sum15 = _mm256_fmadd_ps(a1, b5, sum15);
            }

            // Reduction lambda
            auto hsum = [](const __m256 &v) -> float {
              __m128 lo = _mm256_castps256_ps128(v);
              __m128 hi = _mm256_extractf128_ps(v, 1);
              lo = _mm_add_ps(lo, hi);
              __m128 p = _mm_hadd_ps(lo, lo);
              return _mm_cvtss_f32(_mm_hadd_ps(p, p));
            };

            // Horizontal Sums
            out_ptr0[j] = hsum(sum00);
            out_ptr0[j + 1] = hsum(sum01);
            out_ptr0[j + 2] = hsum(sum02);
            out_ptr0[j + 3] = hsum(sum03);
            out_ptr0[j + 4] = hsum(sum04);
            out_ptr0[j + 5] = hsum(sum05);

            out_ptr1[j] = hsum(sum10);
            out_ptr1[j + 1] = hsum(sum11);
            out_ptr1[j + 2] = hsum(sum12);
            out_ptr1[j + 3] = hsum(sum13);
            out_ptr1[j + 4] = hsum(sum14);
            out_ptr1[j + 5] = hsum(sum15);

            // Handle K tail (remainder)
            for (; k < this->cols; k++) {
              float val_a0 = row_ptr0[k];
              float val_a1 = row_ptr1[k];

              out_ptr0[j] += val_a0 * col_ptr0[k];
              out_ptr0[j + 1] += val_a0 * col_ptr1[k];
              out_ptr0[j + 2] += val_a0 * col_ptr2[k];
              out_ptr0[j + 3] += val_a0 * col_ptr3[k];
              out_ptr0[j + 4] += val_a0 * col_ptr4[k];
              out_ptr0[j + 5] += val_a0 * col_ptr5[k];

              out_ptr1[j] += val_a1 * col_ptr0[k];
              out_ptr1[j + 1] += val_a1 * col_ptr1[k];
              out_ptr1[j + 2] += val_a1 * col_ptr2[k];
              out_ptr1[j + 3] += val_a1 * col_ptr3[k];
              out_ptr1[j + 4] += val_a1 * col_ptr4[k];
              out_ptr1[j + 5] += val_a1 * col_ptr5[k];
            }
          }
          // Clean up remaining J cols
          for (; j < j_block_end; j++) {
            const float *col_ptr = A_data + (j * A.rows);
            float sum0 = 0.0f;
            float sum1 = 0.0f;
            for (int k = 0; k < this->cols; k++) {
              float b_val = col_ptr[k];
              sum0 += row_ptr0[k] * b_val;
              sum1 += row_ptr1[k] * b_val;
            }
            out_ptr0[j] = sum0;
            out_ptr1[j] = sum1;
          }
        }

        // Handle remaining I rows
        for (; i < end_row; i++) {
          const float *row_ptr = this_data + (i * this->cols);
          float *out_ptr = out_data + (i * output.cols);
          for (int j = jc; j < j_block_end; j++) {
            const float *col_ptr = A_data + (j * A.rows);
            __m256 sum_vec = _mm256_setzero_ps();
            int k = 0;
            for (; k <= this->cols - 8; k += 8) {
              sum_vec = _mm256_fmadd_ps(_mm256_loadu_ps(row_ptr + k),
                                        _mm256_loadu_ps(col_ptr + k), sum_vec);
            }
            // Hsum
            __m128 lo = _mm256_castps256_ps128(sum_vec);
            __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
            lo = _mm_add_ps(lo, hi);
            __m128 p = _mm_hadd_ps(lo, lo);
            float sum = _mm_cvtss_f32(_mm_hadd_ps(p, p));

            for (; k < this->cols; k++)
              sum += row_ptr[k] * col_ptr[k];
            out_ptr[j] = sum;
          }
        }
      }
    }
    // Standard Path
    else {
      for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < A.cols; j++) {
          float sum = 0.0f;
          for (int k = 0; k < this->cols; k++) {
            sum += (*this)(i, k) * A(k, j);
          }
          output(i, j) = sum;
        }
      }
    }
  };

  std::vector<std::thread> threads;
  int rows_per_thread = this->rows / num_threads;
  int start = 0;

  for (unsigned int i = 0; i < num_threads; ++i) {
    int end = (i == num_threads - 1) ? this->rows : start + rows_per_thread;
    threads.emplace_back(worker, start, end);
    start = end;
  }

  for (auto &t : threads) {
    t.join();
  }

  return output;
}

bool Matrix::operator==(const Matrix &other) const {
  if (rows != other.rows || cols != other.cols)
    return false;
  for (size_t i = 0; i < elements.size(); ++i) {
    if (std::abs(elements[i] - other.elements[i]) > 1e-5)
      return false;
  }
  return true;
}

void Matrix::print() const {
  std::cout << "Matrix (" << rows << "x" << cols << ") "
            << (rank == row ? "[RowRank]" : "[ColRank]") << ":" << std::endl;
  for (int i = 0; i < rows; i++) {
    std::cout << "  [ ";
    for (int j = 0; j < cols; j++) {
      std::cout << (*this)(i, j) << " ";
    }
    std::cout << "]" << std::endl;
  }
}
