#include "matrix.h"
#include <cassert>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

void test_commutativity() {
  std::cout << "Testing Commutativity (A * B != B * A usually)..." << std::endl;
  // Create two 2x2 matrices
  std::vector<float> dataA = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> dataB = {5.0f, 6.0f, 7.0f, 8.0f};
  Matrix A(2, 2, dataA);
  Matrix B(2, 2, dataB);
  Matrix C(2, 2, dataA, Matrix::column);
  Matrix AB = A * B;

  std::cout << "C:" << std::endl;
  C.print();

  std::cout << "AB:" << std::endl;
  AB.print();

  Matrix CB = C * B;
  std::cout << "CB:" << std::endl;
  CB.print();

  Matrix BA = B * A;
  std::cout << "BA:" << std::endl;
  BA.print();

  if (AB == BA) {
    std::cout << "  Commutativity holds (unexpected for general matrices)!"
              << std::endl;
  } else {
    std::cout << "  Commutativity does NOT hold (as expected)." << std::endl;
  }
}

void test_associativity() {
  std::cout << "Testing Associativity ((A * B) * C == A * (B * C))..."
            << std::endl;
  std::vector<float> dataA = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> dataB = {5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> dataC = {9.0f, 1.0f, 2.0f, 3.0f};

  Matrix A(2, 2, dataA);
  Matrix B(2, 2, dataB);
  Matrix C(2, 2, dataC);

  Matrix AB = A * B;
  Matrix AB_C = AB * C;
  std::cout << "(AB)C:" << std::endl;
  AB_C.print();

  Matrix BC = B * C;
  Matrix A_BC = A * BC;
  std::cout << "A(BC):" << std::endl;
  A_BC.print();

  if (AB_C == A_BC) {
    std::cout << "  Associativity holds." << std::endl;
    assert(false && "Associativity should hold for matrix multiplication");
  }
}

// Helper to generate random matrices with fixed seed for reproducibility
Matrix generate_random_matrix(int rows, int cols, Matrix::Rank rank,
                              int seed = 67) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  std::vector<float> data(rows * cols);
  for (int i = 0; i < rows * cols; ++i) {
    data[i] = dis(gen);
  }
  return Matrix(rows, cols, data, rank);
}

void test_large_multiplication() {
  std::cout << "Testing Large Matrix Multiplication Performance..."
            << std::endl;
  int N = 512; // Size of the matrix (N x N)
  std::cout << "  Matrix Size: " << N << "x" << N << std::endl;

  // Generate matrices
  // A is Row major
  Matrix A = generate_random_matrix(N, N, Matrix::row, 1);
  // B1 is Column major
  Matrix B_col = generate_random_matrix(N, N, Matrix::column, 2);
  // B2 is Row major (same content as B_col ideally, but for performance test
  // just same size/randomness matters)
  Matrix B_row = generate_random_matrix(N, N, Matrix::row, 2);

  // Test 1: Row * Column (Should be faster theoretically due to dot product
  // locality if oiptimized)
  auto start = std::chrono::steady_clock::now();
  Matrix C1 = A * B_col;
  auto end = std::chrono::steady_clock::now();
  auto diff_row_col =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  std::cout << "  Row * Col Time: " << diff_row_col << " ms" << std::endl;

  // Test 2: Row * Row
  start = std::chrono::steady_clock::now();
  Matrix C2 = A * B_row;
  end = std::chrono::steady_clock::now();
  auto diff_row_row =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  std::cout << "  Row * Row Time: " << diff_row_row << " ms" << std::endl;

  if (diff_row_col < diff_row_row) {
    std::cout << "  Row*Col was faster (as expected for efficient dot product "
                 "access)."
              << std::endl;
  } else {
    std::cout << "  Row*Row was faster or equal (unexpected if column-major "
                 "access is optimized)."
              << std::endl;
  }
}

void matrix_tests() {
  std::cout << "Running Matrix Tests..." << std::endl;
  test_commutativity();
  test_associativity();
  test_large_multiplication();
  std::cout << "Tests Completed." << std::endl;
}

int main() {
  auto t = std::chrono::steady_clock::now();
  // tests >>>

  // <<< tests
  auto t_prime = std::chrono::steady_clock::now();
  auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(t_prime - t)
                  .count();
  std::cout << "TotalTime taken: " << diff << " ms" << std::endl;

  return 0;
}