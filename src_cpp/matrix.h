#ifndef MATRIX_H
#define MATRIX_H

#include <cmath>
#include <iostream>
#include <vector>

class Matrix {
public:
  enum Rank { row, column };

private:
  int rows, cols;
  Rank rank;
  std::vector<float> elements;

public:
  Matrix(int m, int n, std::vector<float> x, Rank rank = row);
  Matrix(int m, int n, Rank rank = row);

  int get_rows() const;
  int get_columns() const;

  void transpose();

  // Accessor
  float operator()(int idx_r, int idx_c) const;

  // Mutable accessor for assignment
  float &operator()(int idx_r, int idx_c);

  friend float dot(const std::vector<float> &a, const std::vector<float> &b);

  Matrix operator*(const Matrix &A) const;

  bool operator==(const Matrix &other) const;

  void print() const;
};

#endif // MATRIX_H
