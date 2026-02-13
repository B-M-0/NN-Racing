#include "matrix.h"

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
  // Simple O(N^3) multiplication
  // Result dimensions: (this->rows) x (A.cols)
  Matrix output(this->rows, A.cols, row);

  for (int i = 0; i < this->rows; i++) {
    for (int j = 0; j < A.cols; j++) {
      float sum = 0.0f;
      for (int k = 0; k < this->cols; k++) {
        // Using operator() to handle rank abstraction
        sum += (*this)(i, k) * A(k, j);
      }
      output(i, j) = sum;
    }
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
