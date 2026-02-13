#include "matrix.h"
#include <cassert>
#include <chrono>
#include <iostream>
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
  } else {
    std::cout << "  Associativity FAILED!" << std::endl;
    // In a real test framework we might assert or throw,
    // but for this exercise we print the result.
    assert(false && "Associativity should hold for matrix multiplication");
  }
}

int main() {
  auto t = std::chrono::steady_clock::now();
  std::cout << "Running Matrix Tests..." << std::endl;
  test_commutativity();
  test_associativity();
  std::cout << "Tests Completed." << std::endl;
  auto t_prime = std::chrono::steady_clock::now();
  auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(t_prime - t)
                  .count();
  std::cout << "Time taken: " << diff << " ms" << std::endl;
  return 0;
}